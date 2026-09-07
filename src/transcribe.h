#pragma once
#include <string>
#include <vector>
#include "srt.h"

namespace transcribe {

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
};

// Transcribe mono 16 kHz float PCM into timestamped segments.
bool run(const std::vector<float>& pcm, const Options& opts,
         std::vector<srt::Segment>& out, std::string& err);

} // namespace transcribe
