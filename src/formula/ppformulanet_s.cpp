#include "turbo_ocr/formula/ppformulanet_s.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <format>
#include <iostream>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
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

// Per-RPC wall-clock budget for send/recv; a hung sidecar must not pin a
// Drogon worker forever. Env PPFNS_TIMEOUT_S overrides (default 30s).
int rpc_timeout_s() {
  if (const char *e = std::getenv("PPFNS_TIMEOUT_S")) {
    int v = std::atoi(e);
    if (v > 0) return v;
  }
  return 30;
}

// Arm SO_RCVTIMEO/SO_SNDTIMEO so blocking send/recv return after `secs`
// with EAGAIN/EWOULDBLOCK instead of stalling indefinitely.
void set_socket_timeouts(int fd, int secs) {
  timeval tv{};
  tv.tv_sec = secs;
  tv.tv_usec = 0;
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

// SO_SNDTIMEO/SO_RCVTIMEO bound send/recv but NOT connect(); a wedged peer
// (accept() stalled, backlog full) can hang ::connect() indefinitely on
// AF_UNIX. Drive connect through a non-blocking fd + bounded poll(), then
// restore blocking mode so the existing send/recv timeouts apply. Returns
// true only on a fully established connection. On any path the fd is left in
// blocking mode for the caller.
bool connect_with_timeout(int fd, const sockaddr *addr, socklen_t addrlen,
                          int secs) {
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) return false;
  if (::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) return false;

  bool ok = false;
  int rc = ::connect(fd, addr, addrlen);
  if (rc == 0) {
    ok = true;  // immediate connect (typical for AF_UNIX)
  } else if (errno == EINPROGRESS) {
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLOUT;
    int pr;
    do {
      pr = ::poll(&pfd, 1, secs > 0 ? secs * 1000 : -1);
    } while (pr < 0 && errno == EINTR);
    if (pr > 0) {
      int err = 0;
      socklen_t len = sizeof(err);
      if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) == 0 && err == 0) {
        ok = true;
      }
    }
    // pr == 0 -> timed out; pr < 0 -> poll error; both fall through as failure.
  }

  // Restore blocking mode regardless so SO_SNDTIMEO/SO_RCVTIMEO govern I/O.
  ::fcntl(fd, F_SETFL, flags);
  return ok;
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

// Single-respawn guard: once the shared sidecar has died and been cleared,
// is_ready() reports false for every instance without attempting a fresh
// spawn (we never auto-respawn in-process; a restart re-spawns at boot).
bool g_sidecar_reaped = false;

// Liveness is a property of the process-GLOBAL sidecar, not of any one
// PPFormulaNetS instance. Probe the shared pid under g_sidecar_mu and, if
// the child has exited, clear the global so EVERY instance — not just the
// one that issued the failing RPC — sees the death. Caller must hold
// g_sidecar_mu.
bool global_sidecar_alive_locked() {
  if (g_sidecar_pid <= 0 || g_sidecar_sock_path.empty()) return false;
  int status = 0;
  pid_t r = ::waitpid(g_sidecar_pid, &status, WNOHANG);
  // r == pid: the child exited and we reaped it now.
  // r == -1: it is no longer our child (ECHILD) — already reaped elsewhere or
  // never ours; either way the sidecar we tracked is gone. Treating this as
  // DEAD (rather than the old fall-through to "alive") prevents a permanently
  // stuck alive=true when the pid has been reaped out from under us.
  if (r == g_sidecar_pid || r < 0) {
    if (r < 0) {
      std::cerr << "[PPFormulaNetS] shared sidecar pid=" << g_sidecar_pid
                << " no longer reapable (waitpid errno=" << errno
                << "); marking backend not-ready for all instances\n";
    } else {
      std::cerr << "[PPFormulaNetS] shared sidecar pid=" << g_sidecar_pid
                << " has exited (status=" << status
                << "); marking backend not-ready for all instances\n";
    }
    g_sidecar_pid = -1;
    g_sidecar_sock_path.clear();
    g_sidecar_reaped = true;
    return false;
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

bool PPFormulaNetS::is_ready() const noexcept {
  // Consult the shared sidecar state so every instance observes a death,
  // and reap the global child here if it has exited post-boot.
  std::lock_guard<std::mutex> lk(g_sidecar_mu);
  return global_sidecar_alive_locked();
}

bool PPFormulaNetS::load_model_dir(const std::string &model_dir) {
  // The pipeline passes FORMULA_ONNX, which for formulanet is a *file* path,
  // not a directory; legacy callers still pass a directory. Resolve both: a
  // regular file is used directly (its parent supplies tokenizer.json), a
  // directory is scanned for inference_trt.onnx / inference.onnx.
  fs::path dir;
  std::string model_path;
  std::error_code ec;
  if (fs::is_regular_file(model_dir, ec)) {
    const fs::path file(model_dir);
    dir = file.parent_path();
    const std::string name = file.filename().string();
    if (name.rfind("inference", 0) == 0 && file.extension() == ".onnx") {
      model_path = file.string();
    } else {
      // A file was given but it is not an inference*.onnx — fall back to
      // locating the model alongside it so a stray sibling path still works.
      const auto onnx_patched = (dir / "inference_trt.onnx").string();
      const auto onnx_orig    = (dir / "inference.onnx").string();
      model_path = fs::exists(onnx_patched) ? onnx_patched : onnx_orig;
    }
  } else {
    dir = fs::path(model_dir);
    const auto onnx_patched = (dir / "inference_trt.onnx").string();
    const auto onnx_orig    = (dir / "inference.onnx").string();
    model_path = fs::exists(onnx_patched) ? onnx_patched : onnx_orig;
  }
  if (model_path.empty() || !fs::exists(model_path)) {
    std::cerr << std::format(
        "[PPFormulaNetS] no model resolved from '{}' (expected an "
        "inference*.onnx file or a dir containing inference_trt.onnx / "
        "inference.onnx)", model_dir) << '\n';
    return false;
  }
  const auto tok_path = (dir / "tokenizer.json").string();
  if (!fs::exists(tok_path)) {
    std::cerr << std::format(
        "[PPFormulaNetS] missing tokenizer.json next to model '{}' (dir '{}')",
        model_path, dir.string()) << '\n';
    return false;
  }
  {
    // An explicit load_model_dir (boot or a GpuPipelineEntry::recycle that
    // rebuilds the recognizer) is the legitimate re-establish point: clear the
    // single-respawn latch so a recycle can bring the shared sidecar back if it
    // had previously died. The RPC failure path never reaches here, so it stays
    // no-auto-respawn. If the shared sidecar is still alive, spawn_sidecar
    // adopts it before consulting the latch, so this never double-spawns.
    std::lock_guard<std::mutex> lk(g_sidecar_mu);
    g_sidecar_reaped = false;
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
  if (g_sidecar_reaped) {
    // Single-respawn guard: the shared sidecar died once; we do not
    // resurrect it in-process. A server restart re-spawns it at boot.
    std::cerr << "[PPFormulaNetS] shared sidecar previously died; "
                 "not re-spawning (restart to recover)\n";
    return false;
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

void PPFormulaNetS::reap_if_dead() {
  if (sidecar_pid_ <= 0) return;
  std::lock_guard<std::mutex> lk(g_sidecar_mu);
  // global_sidecar_alive_locked() reaps+clears the shared child (and sets
  // g_sidecar_reaped) if it has exited. Drop this instance's local copy
  // whenever the shared sidecar is gone — including the case where a sibling
  // already reaped it, so the global no longer matches our cached pid.
  if (!global_sidecar_alive_locked() || g_sidecar_pid != sidecar_pid_) {
    sidecar_pid_ = -1;
    sock_path_.clear();
  }
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
  set_socket_timeouts(fd, rpc_timeout_s());

  // Any transport-level failure (connect/send/recv timeout, EOF, EAGAIN)
  // routes here: close the fd, check whether the child has died, and fail
  // the RPC so the caller returns empty formulas rather than hanging.
  auto fail = [&]() -> bool {
    ::close(fd);
    reap_if_dead();
    return false;
  };

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, sock_path_.c_str(), sizeof(addr.sun_path) - 1);
  if (!connect_with_timeout(fd, reinterpret_cast<sockaddr *>(&addr),
                            sizeof(addr), rpc_timeout_s())) {
    return fail();
  }

  uint32_t batch = static_cast<uint32_t>(crops_bgr.size());
  if (!send_all(fd, &batch, sizeof(batch))) return fail();
  for (uint32_t i = 0; i < batch; ++i) {
    uint32_t H = static_cast<uint32_t>(heights[i]);
    uint32_t W = static_cast<uint32_t>(widths[i]);
    if (!send_all(fd, &H, sizeof(H)) ||
        !send_all(fd, &W, sizeof(W)) ||
        !send_all(fd, crops_bgr[i].data(), crops_bgr[i].size())) {
      return fail();
    }
  }

  uint32_t status = 0, count = 0;
  if (!recv_all(fd, &status, sizeof(status)) ||
      !recv_all(fd, &count, sizeof(count))) {
    return fail();
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
    if (!recv_all(fd, &llen, sizeof(llen))) return fail();
    std::string s(llen, '\0');
    if (llen > 0 && !recv_all(fd, s.data(), llen)) return fail();
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
    // Fail LOUD (no-silent-failure contract): a configured formula stage just
    // produced nothing for these regions because the sidecar RPC failed (OOM /
    // crash / timeout), NOT because the regions had no formula. is_ready() flips
    // to false on sidecar death so /capabilities + health surface the
    // degradation; this line makes the per-request loss un-missable in logs.
    std::cerr << "[PPFormulaNetS] ERROR: formula RPC failed — " << boxes.size()
              << " formula region(s) DROPPED (sidecar degraded: check VRAM / "
                 "the ppformulanet_s sidecar process). is_ready now="
              << (is_ready() ? "true" : "false") << '\n';
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
