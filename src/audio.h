#pragma once
#include <string>
#include <vector>

namespace audio {

struct DecodeOptions {
    int sample_rate   = 16000; // whisper expects 16 kHz
    int stream_index  = -1;    // -1 = best audio stream
    double max_seconds = 0.0;  // 0 = whole file; else stop after N seconds (from start)
};

// Decode the audio of any media file to mono float32 PCM at opts.sample_rate.
// Returns false (and sets err) on failure.
bool decode(const std::string& path, const DecodeOptions& opts,
            std::vector<float>& out_pcm, std::string& err);

} // namespace audio
