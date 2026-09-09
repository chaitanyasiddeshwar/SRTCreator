#pragma once
#include <functional>
#include <string>
#include <vector>
#include "srt.h"

struct whisper_context;

namespace transcribe {

// A VAD-detected speech region, in original-timeline seconds.
struct VadRegion { double t0 = 0.0; double t1 = 0.0; };

// A compute device discovered by ggml's dynamic backend loader
// (ggml_backend_load_all over the backend DLLs next to the exe).
struct DeviceInfo {
    std::string name;         // ggml device name, e.g. "CUDA0", "Vulkan0", "CPU"
    std::string description;  // human string, e.g. "NVIDIA GeForce RTX 3080 Ti"
    std::string backend;      // registered backend/reg name, e.g. "CUDA", "Vulkan", "CPU"
    bool is_gpu = false;
};

// Load the backend DLLs (once) and enumerate the compute devices ggml found.
// Drives the CLI/GUI "which backend am I on" reporting and the GUI device picker.
// GPU devices come first, then CPU. Empty only if even the CPU backend is missing.
std::vector<DeviceInfo> available_devices();

struct Options {
    std::string model_path;             // resolved local model file
    std::string language = "auto";      // ISO code or "auto"
    bool translate = false;             // translate to English
    bool flash_attn = true;             // ggml fused flash attention
    bool vad = false;                   // enable Silero VAD (needs vad_model_path)
    std::string vad_model_path;
    int  threads = 0;                   // 0 = auto
    bool word_timestamps = false;
    bool verbose = false;

    // Compute device selection. -1 = auto (best available GPU, else CPU). Otherwise
    // an index into available_devices(): a GPU entry forces that GPU (CUDA/Vulkan),
    // the CPU entry forces CPU. Drives the GUI backend picker.
    int  device_index = -1;

    // Optional live callbacks (used by the GUI; left null by the CLI).
    // on_segment fires as each new segment is decoded; on_progress reports 0-100.
    // Both fire on the worker/whisper thread - marshal to your UI thread.
    std::function<void(const srt::Segment&)> on_segment;
    std::function<void(int)>                 on_progress;

    // Optional debug output: if set, filled with the VAD speech regions whisper
    // used (original timeline). Null unless --debug.
    std::vector<VadRegion>* vad_regions = nullptr;
};

// Reusable Whisper session to keep the model in GPU VRAM across many per-region
// transcriptions (segment mode slices the track into speech regions and runs each
// through the same context) without repeatedly reloading multi-gigabyte models
// from disk and thrashing CUDA.
class Session {
public:
    Session();
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&& other) noexcept;
    Session& operator=(Session&& other) noexcept;

    // Initialize context and load model weights to GPU once
    bool init(const Options& opts, std::string& err);

    // Transcribe audio using the already-loaded model context
    bool transcribe(const std::vector<float>& pcm, const Options& opts,
                    std::vector<srt::Segment>& out, std::string& err);

    // Explicitly release model, GPU buffers, and reset CUDA device memory
    void close();

    bool is_valid() const;

private:
    ::whisper_context* ctx_ = nullptr;
};

// Transcribe mono 16 kHz float PCM into timestamped segments (one-shot).
bool run(const std::vector<float>& pcm, const Options& opts,
         std::vector<srt::Segment>& out, std::string& err);

} // namespace transcribe
