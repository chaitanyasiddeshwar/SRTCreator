// segment_mode.{h,cpp} - the transcription pipeline.
//
// Instead of handing the whole audio to whisper with its internal Silero VAD
// (which excises silence and *concatenates* speech into 30s mel windows, causing
// the seam / bunching / cascade timing pathology described in CLAUDE.md 3.1), this
// pipeline:
//   1. runs Silero VAD standalone over the full 16k mono PCM to partition it into
//      speech regions (original timeline), and
//   2. transcribes each region in isolation via the already-loaded Whisper
//      Session with whisper's own VAD OFF, offsetting timestamps back by the
//      region start.
// No concatenation => no seam => cue timing is just region_start + local time.
//
// It is intentionally self-contained and shallow: it reuses transcribe::Session
// and produces a plain segment vector for srt::write.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "srt.h"
#include "transcribe.h"

namespace segment_mode {

struct Options {
    std::string vad_model_path;      // Silero VAD ggml model (same file the default path uses)

    // Silero params. Threshold is lowered vs the whisper default (0.5) because film
    // dialogue over score/effects is often quiet; we would rather over-detect and
    // let whisper emit nothing on a false region than miss real speech (there is no
    // whisper-decode fallback in this mode - a dropped region is lost).
    float threshold        = 0.30f;
    int   min_speech_ms    = 150;
    int   min_silence_ms   = 200;
    float max_speech_s     = 20.0f;  // cap a region so a slice never exceeds ~one mel window
    int   speech_pad_ms    = 200;

    // Extra raw-PCM context spliced around each VAD region before transcription, so
    // whisper sees a little lead-in/out and does not clip word edges. Timestamps are
    // still offset by the padded slice start, then DTW (if word_timestamps) tightens.
    double slice_pad_s     = 0.20;

    // Silero VAD runs on CPU: it is tiny (~0.9 MB) and whisper's own internal VAD
    // path uses use_gpu=false too. Forcing it onto CUDA0 alongside the whisper
    // session crashes this build ("no GPU found" -> pre-allocated tensor assert).
    bool use_gpu    = false;
    int  gpu_device = 0;
    int  threads    = 0;             // 0 = auto (for the VAD context)
};

// Run the alternate pipeline. `session` must already be init()'d (model in VRAM);
// `base_topts` supplies model/language/translate/word-timestamp settings for the
// per-slice transcription (its vad flag is ignored - VAD is forced off per slice).
// Recovered cues are appended to `out` in chronological order.
//
// Callbacks (all optional, fire on the calling/worker thread):
//   on_log      - textual milestones (VAD region count, final summary)
//   on_progress - 0..100 as speech regions are transcribed (drives a progress bar)
//   on_cue      - each emitted cue, in original-timeline order (live transcript)
// Per-slice whisper callbacks are intentionally suppressed; progress/cues are
// reported at region granularity instead so the UI updates without callback spam.
bool run(const std::vector<float>& pcm,
         transcribe::Session& session,
         const transcribe::Options& base_topts,
         const Options& opts,
         std::vector<srt::Segment>& out,
         std::string& err,
         std::function<void(const std::string&)> on_log = nullptr,
         std::function<void(int)> on_progress = nullptr,
         std::function<void(const srt::Segment&)> on_cue = nullptr);

} // namespace segment_mode
