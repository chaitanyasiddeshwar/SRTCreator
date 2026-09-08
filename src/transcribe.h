#pragma once
#include <functional>
#include <string>
#include <vector>
#include "srt.h"

namespace transcribe {

// A VAD-detected speech region, in original-timeline seconds.
struct VadRegion { double t0 = 0.0; double t1 = 0.0; };

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

    // Optional live callbacks (used by the GUI; left null by the CLI).
    // on_segment fires as each new segment is decoded; on_progress reports 0-100.
    // Both fire on the worker/whisper thread - marshal to your UI thread.
    std::function<void(const srt::Segment&)> on_segment;
    std::function<void(int)>                 on_progress;

    // Optional debug output: if set, filled with the VAD speech regions whisper
    // used (original timeline). Null unless --debug.
    std::vector<VadRegion>* vad_regions = nullptr;
};

// Transcribe mono 16 kHz float PCM into timestamped segments.
bool run(const std::vector<float>& pcm, const Options& opts,
         std::vector<srt::Segment>& out, std::string& err);

} // namespace transcribe
