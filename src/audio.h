#pragma once
#include <functional>
#include <string>
#include <vector>

namespace audio {

struct DecodeOptions {
    int sample_rate   = 16000; // 16 kHz for whisper; 44100 for separation
    int channels      = 1;     // 1 = mono; 2 = stereo (separation model input)
    int stream_index  = -1;    // -1 = best audio stream
    double max_seconds = 0.0;  // 0 = whole file; else stop after N seconds (from start)

    // If true and the source has a Front-Center channel (5.1/7.1), take ONLY that
    // channel (movie dialogue lives there) instead of downmixing all channels.
    // Auto-falls back to a normal downmix when there is no center channel.
    bool center_channel_only = false;

    // Optional 0-100 progress during decode (best-effort; needs a known duration).
    std::function<void(int)> on_progress;
};

// Decode the audio of any media file to float32 PCM (interleaved, opts.channels)
// at opts.sample_rate. Returns false (and sets err) on failure.
bool decode(const std::string& path, const DecodeOptions& opts,
            std::vector<float>& out_pcm, std::string& err);

// Resample an in-memory interleaved float buffer to mono at out_rate.
bool resample_to_mono(const std::vector<float>& in, int in_rate, int in_channels,
                      int out_rate, std::vector<float>& out, std::string& err);

} // namespace audio
