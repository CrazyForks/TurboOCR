#include "turbo_ocr/vlm/crop_pool.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "simdutf.h"

namespace turbo_ocr::vlm {

namespace {

int get_env_int(const char *k, int dflt) {
    const char *v = std::getenv(k);
    if (!v || !v[0]) return dflt;
    try { return std::stoi(v); } catch (...) { return dflt; }
}

// Build JSON body for the chat/completions call.
std::string build_json_body(const std::vector<uint8_t> &png_bytes,
                             const std::string &prompt,
                             const std::string &model,
                             int max_tokens) {
    // Base64-encode the PNG.
    size_t b64_len = simdutf::base64_length_from_binary(png_bytes.size());
    std::string b64(b64_len, '\0');
    simdutf::binary_to_base64(
        reinterpret_cast<const char *>(png_bytes.data()),
        png_bytes.size(), b64.data());

    // Build content array for the user message.
    nlohmann::json image_block;
    image_block["type"] = "image_url";
    image_block["image_url"] = nlohmann::json::object();
    image_block["image_url"]["url"] = std::string("data:image/png;base64,") + b64;

    nlohmann::json text_block;
    text_block["type"] = "text";
    text_block["text"] = prompt;

    nlohmann::json content = nlohmann::json::array();
    content.push_back(std::move(image_block));
    content.push_back(std::move(text_block));

    nlohmann::json user_msg;
    user_msg["role"] = "user";
    user_msg["content"] = std::move(content);

    nlohmann::json messages = nlohmann::json::array();
    messages.push_back(std::move(user_msg));

    nlohmann::json body;
    body["model"] = model;
    body["max_tokens"] = max_tokens;
    body["temperature"] = 0.0;
    body["messages"] = std::move(messages);

    return body.dump();
}

// Extract the assistant's text content from a chat/completions response.
std::string parse_content(const std::string &body) {
    try {
        auto j = nlohmann::json::parse(body);
        return j.at("choices").at(0).at("message").at("content")
                 .get<std::string>();
    } catch (...) {
        return {};
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

VLMCropPool &VLMCropPool::instance() {
    static VLMCropPool pool;
    return pool;
}

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

VLMCropPool::VLMCropPool()
    : max_concurrency_(std::max(1, get_env_int("VLM_GLOBAL_CONCURRENCY", 50))) {
    // Pipe for waking the worker when new items are queued.
    // Both ends must be non-blocking: the reader uses drain_pipe() which loops
    // until EAGAIN, and the writer uses write() from submit() which must not
    // stall if the pipe buffer fills.
    int fds[2];
    if (pipe(fds) != 0) {
        std::cerr << "[VLMCropPool] pipe() failed: " << strerror(errno) << '\n';
    } else {
        wake_rd_ = fds[0];
        wake_wr_ = fds[1];
        fcntl(wake_rd_, F_SETFL, fcntl(wake_rd_, F_GETFL, 0) | O_NONBLOCK);
        fcntl(wake_wr_, F_SETFL, fcntl(wake_wr_, F_GETFL, 0) | O_NONBLOCK);
    }

    multi_ = curl_multi_init();
    if (!multi_) {
        std::cerr << "[VLMCropPool] curl_multi_init() failed\n";
        return;
    }
    // Limit open connections to avoid FD exhaustion; each handle is loopback
    // so one persistent connection per handle is fine.
    curl_multi_setopt(multi_, CURLMOPT_MAXCONNECTS, (long)max_concurrency_);

    worker_ = std::thread([this] { worker_loop(); });
    std::cout << "[VLMCropPool] started max_concurrency=" << max_concurrency_
              << " wake_pipe=" << wake_rd_ << "/" << wake_wr_ << '\n';
    std::cout.flush();
}

VLMCropPool::~VLMCropPool() {
    shutdown();
}

void VLMCropPool::shutdown() {
    bool expected = false;
    if (!stop_.compare_exchange_strong(expected, true)) return;

    // Wake the worker.
    if (wake_wr_ >= 0) { char b = 0; (void)write(wake_wr_, &b, 1); }
    if (worker_.joinable()) worker_.join();

    if (wake_rd_ >= 0) { close(wake_rd_); wake_rd_ = -1; }
    if (wake_wr_ >= 0) { close(wake_wr_); wake_wr_ = -1; }

    // Drain any remaining pending items (set_value empty string).
    std::lock_guard<std::mutex> lk(queue_mu_);
    while (!queue_.empty()) {
        try { queue_.front()->result.set_value({}); } catch (...) {}
        queue_.pop();
    }

    if (multi_) {
        curl_multi_cleanup(multi_);
        multi_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// submit()
// ---------------------------------------------------------------------------

std::future<std::string> VLMCropPool::submit(std::vector<uint8_t> png_bytes,
                                              std::string          prompt,
                                              std::string          model,
                                              int                  max_tokens,
                                              int                  timeout_s,
                                              std::string          base_url) {
    auto req = std::make_unique<CropRequest>();
    req->png_bytes  = std::move(png_bytes);
    req->prompt     = std::move(prompt);
    req->model      = std::move(model);
    req->max_tokens = max_tokens;
    req->timeout_s  = timeout_s;
    req->base_url   = std::move(base_url);
    auto fut = req->result.get_future();

    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        queue_.push(std::move(req));
    }
    // Wake worker.
    if (wake_wr_ >= 0) { char b = 0; (void)write(wake_wr_, &b, 1); }
    queue_cv_.notify_one();
    return fut;
}

// ---------------------------------------------------------------------------
// Worker thread: curl_multi_poll loop
// ---------------------------------------------------------------------------

size_t VLMCropPool::write_cb(char *ptr, size_t sz, size_t nmemb, void *ud) {
    auto *ctx = static_cast<HandleCtx *>(ud);
    ctx->response.append(ptr, sz * nmemb);
    return sz * nmemb;
}

void VLMCropPool::complete_handle(CURL *easy, CURLcode rc) {
    HandleCtx *ctx = nullptr;
    curl_easy_getinfo(easy, CURLINFO_PRIVATE, &ctx);
    if (!ctx) {
        curl_multi_remove_handle(multi_, easy);
        curl_easy_cleanup(easy);
        return;
    }

    std::string result;
    if (rc == CURLE_OK) {
        long status = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
        if (status >= 200 && status < 300) {
            result = parse_content(ctx->response);
        } else {
            std::cerr << "[VLMCropPool] HTTP " << status
                      << " body=" << ctx->response.substr(0, 200) << '\n';
        }
    } else {
        std::cerr << "[VLMCropPool] curl error: " << curl_easy_strerror(rc) << '\n';
    }

    try { ctx->result.set_value(std::move(result)); } catch (...) {}

    curl_multi_remove_handle(multi_, easy);
    curl_slist_free_all(ctx->headers);
    curl_easy_cleanup(easy);
    delete ctx;

    in_flight_.fetch_sub(1, std::memory_order_relaxed);
}

void VLMCropPool::try_dispatch() {
    // Pop as many items as we have free capacity.
    while (in_flight_.load(std::memory_order_relaxed) < max_concurrency_) {
        std::unique_ptr<CropRequest> req;
        {
            std::lock_guard<std::mutex> lk(queue_mu_);
            if (queue_.empty()) break;
            req = std::move(queue_.front());
            queue_.pop();
        }

        // Build the JSON body.
        std::string json_body;
        try {
            json_body = build_json_body(req->png_bytes, req->prompt,
                                        req->model, req->max_tokens);
        } catch (const std::exception &e) {
            std::cerr << "[VLMCropPool] json build error: " << e.what() << '\n';
            try { req->result.set_value({}); } catch (...) {}
            continue;
        }

        auto *ctx       = new HandleCtx();
        ctx->post_body  = std::move(json_body);
        ctx->result     = std::move(req->result);
        ctx->pool       = this;

        CURL *easy = curl_easy_init();
        if (!easy) {
            std::cerr << "[VLMCropPool] curl_easy_init() failed\n";
            try { ctx->result.set_value({}); } catch (...) {}
            delete ctx;
            continue;
        }
        ctx->easy = easy;

        ctx->headers = nullptr;
        ctx->headers = curl_slist_append(ctx->headers, "Content-Type: application/json");
        ctx->headers = curl_slist_append(ctx->headers, "Accept: application/json");
        // Keep-alive on loopback: no cost, avoids TCP setup per request.
        ctx->headers = curl_slist_append(ctx->headers, "Connection: keep-alive");

        std::string url = req->base_url + "/v1/chat/completions";
        curl_easy_setopt(easy, CURLOPT_URL,           url.c_str());
        curl_easy_setopt(easy, CURLOPT_POST,          1L);
        curl_easy_setopt(easy, CURLOPT_POSTFIELDS,    ctx->post_body.c_str());
        curl_easy_setopt(easy, CURLOPT_POSTFIELDSIZE, (long)ctx->post_body.size());
        curl_easy_setopt(easy, CURLOPT_HTTPHEADER,    ctx->headers);
        curl_easy_setopt(easy, CURLOPT_TIMEOUT,       (long)req->timeout_s);
        curl_easy_setopt(easy, CURLOPT_NOSIGNAL,      1L);
        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(easy, CURLOPT_WRITEDATA,     ctx);
        curl_easy_setopt(easy, CURLOPT_PRIVATE,       ctx);
        curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
        // HTTP/1.1 keep-alive is enough for loopback.
        curl_easy_setopt(easy, CURLOPT_HTTP_VERSION,  CURL_HTTP_VERSION_1_1);

        curl_multi_add_handle(multi_, easy);
        in_flight_.fetch_add(1, std::memory_order_relaxed);
    }
}

void VLMCropPool::worker_loop() {
    // Drain the wake pipe without blocking.
    auto drain_pipe = [this]() {
        if (wake_rd_ < 0) return;
        char buf[64];
        while (read(wake_rd_, buf, sizeof(buf)) > 0) {}
    };

    while (!stop_.load(std::memory_order_relaxed)) {
        // Dispatch any queued items up to concurrency limit.
        try_dispatch();

        int running = 0;
        CURLMcode mc = curl_multi_perform(multi_, &running);
        if (mc != CURLM_OK && mc != CURLM_CALL_MULTI_PERFORM) {
            std::cerr << "[VLMCropPool] curl_multi_perform error: "
                      << curl_multi_strerror(mc) << '\n';
        }

        // Harvest completed transfers.
        int msgs_in_queue = 0;
        CURLMsg *msg;
        while ((msg = curl_multi_info_read(multi_, &msgs_in_queue)) != nullptr) {
            if (msg->msg == CURLMSG_DONE) {
                complete_handle(msg->easy_handle, msg->data.result);
                // After completing, we may have new capacity — try dispatching.
                try_dispatch();
            }
        }

        // Poll with a short timeout; the wake fd lets us wake early.
        // Use curl_multi_poll if available (libcurl ≥7.68); fall back to
        // curl_multi_wait otherwise.
        int numfds = 0;
        struct curl_waitfd extra_fd;
        extra_fd.fd      = wake_rd_;
        extra_fd.events  = CURL_WAIT_POLLIN;
        extra_fd.revents = 0;

        // 200 ms timeout — fast enough to catch new items without spinning.
        curl_multi_poll(multi_, (wake_rd_ >= 0 ? &extra_fd : nullptr),
                        (wake_rd_ >= 0 ? 1u : 0u), 200, &numfds);

        if (extra_fd.revents & CURL_WAIT_POLLIN) drain_pipe();
    }

    // Drain remaining handles on shutdown.
    int running = 0;
    do {
        curl_multi_perform(multi_, &running);
        int msgs = 0;
        CURLMsg *msg;
        while ((msg = curl_multi_info_read(multi_, &msgs)) != nullptr) {
            if (msg->msg == CURLMSG_DONE)
                complete_handle(msg->easy_handle, msg->data.result);
        }
        if (running > 0) curl_multi_poll(multi_, nullptr, 0, 50, nullptr);
    } while (running > 0);
}

} // namespace turbo_ocr::vlm
