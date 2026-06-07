#include "turbo_ocr/formula/ppformulanet_s.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "turbo_ocr/common/cuda_check.h"

namespace fs = std::filesystem;

namespace turbo_ocr::formula {

namespace {

// Multiple PPFormulaNetS instances (e.g. one per worker pipeline in a
// pool) share a single Python ORT sidecar so the 300 MB model is loaded
// once. The first instance to call load_model_dir spawns it; subsequent
// instances skip the spawn and just connect to the existing socket.
std::mutex  g_sidecar_mu;
pid_t       g_sidecar_pid = -1;
std::string g_sidecar_sock_path;

// Default path inside the docker image; overridable via env for host tests.
std::string default_sidecar_script() {
  if (const char *e = std::getenv("PPFNS_SIDECAR_SCRIPT")) {
    return std::string(e);
  }
  return "/app/scripts/ppformulanet_s_sidecar.py";
}

std::string default_sock_path() {
  if (const char *e = std::getenv("PPFNS_SOCK")) return std::string(e);
  return "/tmp/ppformulanet_s.sock";
}

// Tight blocking sendall on a SOCK_STREAM socket.
bool send_all(int fd, const void *buf, size_t n) {
  const char *p = static_cast<const char *>(buf);
  while (n > 0) {
    ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
    if (w <= 0) {
      if (w < 0 && errno == EINTR) continue;
      return false;
    }
    p += w; n -= static_cast<size_t>(w);
  }
  return true;
}

bool recv_all(int fd, void *buf, size_t n) {
  char *p = static_cast<char *>(buf);
  while (n > 0) {
    ssize_t r = ::recv(fd, p, n, 0);
    if (r <= 0) {
      if (r < 0 && errno == EINTR) continue;
      return false;
    }
    p += r; n -= static_cast<size_t>(r);
  }
  return true;
}

} // namespace

PPFormulaNetS::PPFormulaNetS() = default;

PPFormulaNetS::~PPFormulaNetS() noexcept {
  // Sidecar is shared across instances; only the owner that spawned it
  // is allowed to tear it down. For now we leave it alive until process
  // exit (no atexit hook) — the kernel reaps the orphan when the parent
  // turboocr-server exits.
}

bool PPFormulaNetS::load_model_dir(const std::string &model_dir) {
  const fs::path dir(model_dir);
  const auto onnx_patched = (dir / "inference_trt.onnx").string();
  const auto onnx_orig    = (dir / "inference.onnx").string();
  const std::string model_path =
      fs::exists(onnx_patched) ? onnx_patched : onnx_orig;
  if (!fs::exists(model_path)) {
    std::cerr << std::format(
        "[PPFormulaNetS] no model in dir '{}' (looked for inference_trt.onnx "
        "+ inference.onnx)", model_dir) << '\n';
    return false;
  }
  const auto tok_path = (dir / "tokenizer.json").string();
  if (!fs::exists(tok_path)) {
    std::cerr << std::format(
        "[PPFormulaNetS] missing tokenizer.json in dir '{}'", model_dir) << '\n';
    return false;
  }
  return spawn_sidecar(model_path, tok_path);
}

bool PPFormulaNetS::load_tokenizer(const std::string & /*path*/) {
  // PPFormulaNetS sidecar loads its own tokenizer from the model dir.
  return true;
}

bool PPFormulaNetS::spawn_sidecar(const std::string &model_path,
                                   const std::string &tokenizer_path) {
  std::lock_guard<std::mutex> lk(g_sidecar_mu);
  if (g_sidecar_pid > 0 && !g_sidecar_sock_path.empty()) {
    // Already running — adopt the existing socket; first owner keeps PID.
    sock_path_ = g_sidecar_sock_path;
    sidecar_pid_ = g_sidecar_pid;
    return true;
  }

  std::string sock = default_sock_path();
  ::unlink(sock.c_str());

  std::string script = default_sidecar_script();
  if (!fs::exists(script)) {
    std::cerr << "[PPFormulaNetS] sidecar script missing: " << script << '\n';
    return false;
  }

  std::vector<std::string> args_s = {
      "python3", script,
      "--model", model_path,
      "--tokenizer", tokenizer_path,
      "--socket", sock,
  };
  std::vector<char *> argv;
  argv.reserve(args_s.size() + 1);
  for (auto &s : args_s) argv.push_back(const_cast<char *>(s.c_str()));
  argv.push_back(nullptr);

  pid_t pid = -1;
  int rc = ::posix_spawnp(&pid, "python3", nullptr, nullptr,
                           argv.data(), environ);
  if (rc != 0) {
    std::cerr << "[PPFormulaNetS] posix_spawnp python3 failed: "
              << std::strerror(rc) << '\n';
    return false;
  }
  sock_path_ = sock;
  sidecar_pid_ = pid;
  g_sidecar_pid = pid;
  g_sidecar_sock_path = sock;
  std::cerr << "[PPFormulaNetS] sidecar spawned pid=" << pid << '\n';

  if (!wait_socket_ready()) {
    std::cerr << "[PPFormulaNetS] sidecar did not become ready\n";
    ::kill(sidecar_pid_, SIGTERM);
    ::waitpid(sidecar_pid_, nullptr, 0);
    sidecar_pid_ = -1;
    g_sidecar_pid = -1;
    g_sidecar_sock_path.clear();
    return false;
  }
  std::cerr << "[PPFormulaNetS] sidecar ready on " << sock_path_ << '\n';
  return true;
}

bool PPFormulaNetS::wait_socket_ready() {
  // Sidecar must load the 300 MB ONNX + tokenizer; allow up to 180s.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(180);
  while (std::chrono::steady_clock::now() < deadline) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd >= 0) {
      sockaddr_un addr{};
      addr.sun_family = AF_UNIX;
      std::strncpy(addr.sun_path, sock_path_.c_str(), sizeof(addr.sun_path) - 1);
      if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
        ::close(fd);
        return true;
      }
      ::close(fd);
    }
    // Detect crash early.
    int status = 0;
    pid_t r = ::waitpid(sidecar_pid_, &status, WNOHANG);
    if (r == sidecar_pid_) {
      std::cerr << "[PPFormulaNetS] sidecar exited during boot status=" << status << '\n';
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
  return false;
}

bool PPFormulaNetS::rpc_call(
    const std::vector<std::vector<uint8_t>> &crops_bgr,
    const std::vector<int> &widths,
    const std::vector<int> &heights,
    std::vector<std::string> &out_latex) {
  out_latex.clear();
  if (crops_bgr.empty()) return true;

  std::lock_guard<std::mutex> lk(rpc_mu_);
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return false;
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, sock_path_.c_str(), sizeof(addr.sun_path) - 1);
  if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return false;
  }

  uint32_t batch = static_cast<uint32_t>(crops_bgr.size());
  if (!send_all(fd, &batch, sizeof(batch))) { ::close(fd); return false; }
  for (uint32_t i = 0; i < batch; ++i) {
    uint32_t H = static_cast<uint32_t>(heights[i]);
    uint32_t W = static_cast<uint32_t>(widths[i]);
    if (!send_all(fd, &H, sizeof(H)) ||
        !send_all(fd, &W, sizeof(W)) ||
        !send_all(fd, crops_bgr[i].data(), crops_bgr[i].size())) {
      ::close(fd);
      return false;
    }
  }

  uint32_t status = 0, count = 0;
  if (!recv_all(fd, &status, sizeof(status)) ||
      !recv_all(fd, &count, sizeof(count))) {
    ::close(fd);
    return false;
  }
  if (status != 0) {
    uint32_t mlen = count;
    std::string msg(mlen, '\0');
    if (mlen > 0) recv_all(fd, msg.data(), mlen);
    std::cerr << "[PPFormulaNetS] sidecar error: " << msg << '\n';
    ::close(fd);
    return false;
  }
  out_latex.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t llen = 0;
    if (!recv_all(fd, &llen, sizeof(llen))) { ::close(fd); return false; }
    std::string s(llen, '\0');
    if (llen > 0 && !recv_all(fd, s.data(), llen)) { ::close(fd); return false; }
    out_latex.push_back(std::move(s));
  }
  ::close(fd);
  return true;
}

std::vector<FormulaEngineResult>
PPFormulaNetS::run(const GpuImage &page, const std::vector<Box> &boxes,
                    cudaStream_t stream) {
  std::vector<FormulaEngineResult> out;
  if (boxes.empty()) return out;
  if (page.empty()) {
    std::cerr << "[PPFormulaNetS] empty page\n";
    return out;
  }
  if (!is_ready()) {
    std::cerr << "[PPFormulaNetS] not ready\n";
    return out;
  }

  const size_t need = static_cast<size_t>(page.rows) * page.step;
  if (host_page_.size() < need) host_page_.resize(need);
  // Copy page from GPU to host (whole page once; cheap, ~5MB for A4 BGR).
  if (cudaSuccess !=
      cudaMemcpyAsync(host_page_.data(), page.data, need,
                      cudaMemcpyDeviceToHost, stream)) {
    std::cerr << "[PPFormulaNetS] page D2H failed\n";
    return out;
  }
  cudaStreamSynchronize(stream);

  std::vector<std::vector<uint8_t>> crops;
  std::vector<int> widths, heights;
  crops.reserve(boxes.size());
  widths.reserve(boxes.size());
  heights.reserve(boxes.size());
  for (const auto &b : boxes) {
    auto r = aabb(b);
    int x0 = std::clamp(r[0], 0, std::max(0, page.cols - 1));
    int y0 = std::clamp(r[1], 0, std::max(0, page.rows - 1));
    int x1 = std::clamp(r[2], x0, page.cols);
    int y1 = std::clamp(r[3], y0, page.rows);
    int w = std::max(1, x1 - x0);
    int h = std::max(1, y1 - y0);
    std::vector<uint8_t> crop(static_cast<size_t>(w) * h * 3);
    const uint8_t *src = host_page_.data() + static_cast<size_t>(y0) * page.step
                          + static_cast<size_t>(x0) * 3;
    for (int row = 0; row < h; ++row) {
      std::memcpy(crop.data() + static_cast<size_t>(row) * w * 3,
                  src + static_cast<size_t>(row) * page.step,
                  static_cast<size_t>(w) * 3);
    }
    crops.push_back(std::move(crop));
    widths.push_back(w);
    heights.push_back(h);
  }

  std::vector<std::string> latex_results;
  if (!rpc_call(crops, widths, heights, latex_results)) {
    std::cerr << "[PPFormulaNetS] RPC failed; returning empty formulas\n";
    out.resize(boxes.size());
    return out;
  }
  out.reserve(latex_results.size());
  for (auto &s : latex_results) {
    FormulaEngineResult r;
    r.latex = std::move(s);
    r.token_count = r.latex.size();  // approximation; sidecar doesn't return n_tokens
    r.hit_eos = true;
    out.push_back(std::move(r));
  }
  // Pad if RPC under-returned (shouldn't happen).
  while (out.size() < boxes.size()) out.push_back({});
  return out;
}

} // namespace turbo_ocr::formula
