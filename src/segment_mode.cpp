#include "segment_mode.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <thread>

#include "whisper.h"

namespace segment_mode {

namespace {
constexpr int kSampleRate = 16000;

// --- Precision helpers (segment-mode only; do not affect the default pipeline) ---

bool has_alnum(const std::string& s) {
    for (unsigned char c : s) if (std::isalnum(c)) return true;
    return false;
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> w;
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!cur.empty()) { w.push_back(cur); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
}

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Collapse a fully-periodic repeated phrase to one instance. Targets whisper's
// short-slice hallucination pattern where an utterance is emitted 2+ times, e.g.
// "You're a fool. You're a fool." -> "You're a fool.". Conservative: a multi-word
// period must repeat >=2x, a single-word period >=3x, so legitimate emphatic
// repetition like "No, no, no." (differing punctuation) or a doubled word survives.
// Case-insensitive match; original casing/punctuation of the first instance kept.
std::string collapse_repeat(const std::string& text) {
    std::vector<std::string> w = split_ws(text);
    const size_t n = w.size();
    if (n < 2) return text;
    std::vector<std::string> lw(n);
    for (size_t i = 0; i < n; ++i) lw[i] = lower(w[i]);
    for (size_t p = 1; p <= n / 2; ++p) {
        if (n % p) continue;
        size_t reps = n / p;
        if (p == 1 ? reps < 3 : reps < 2) continue;
        bool periodic = true;
        for (size_t i = p; i < n && periodic; ++i)
            if (lw[i] != lw[i - p]) periodic = false;
        if (periodic) {
            std::string out;
            for (size_t i = 0; i < p; ++i) { if (i) out.push_back(' '); out += w[i]; }
            return out;
        }
    }
    return text;
}

// Returns the cleaned cue text, or empty string if the cue should be dropped.
std::string clean_cue_text(const std::string& raw) {
    std::string t = collapse_repeat(trim(raw));
    if (!has_alnum(t)) return std::string(); // empty / punctuation-only
    return t;
}

// Small RAII wrappers so early returns can't leak the standalone VAD context/segs.
struct VadCtxGuard {
    whisper_vad_context* p = nullptr;
    ~VadCtxGuard() { if (p) whisper_vad_free(p); }
};
struct VadSegsGuard {
    whisper_vad_segments* p = nullptr;
    ~VadSegsGuard() { if (p) whisper_vad_free_segments(p); }
};
} // namespace

bool run(const std::vector<float>& pcm,
         transcribe::Session& session,
         const transcribe::Options& base_topts,
         const Options& opts,
         std::vector<srt::Segment>& out,
         std::string& err,
         std::function<void(const std::string&)> on_log,
         std::function<void(int)> on_progress,
         std::function<void(const srt::Segment&)> on_cue) {
    auto log = [&](const std::string& m) { if (on_log) on_log(m); };

    if (!session.is_valid()) { err = "segment_mode: session not initialized"; return false; }
    if (pcm.empty()) return true;
    if (opts.vad_model_path.empty()) { err = "segment_mode: no VAD model path"; return false; }

    const double total_sec = (double)pcm.size() / kSampleRate;

    // --- Standalone Silero VAD over the whole track (original timeline) ---
    whisper_vad_context_params cp = whisper_vad_default_context_params();
    cp.use_gpu    = opts.use_gpu;
    cp.gpu_device = opts.gpu_device;
    cp.n_threads  = opts.threads > 0 ? opts.threads
                    : (int)std::max(1u, std::thread::hardware_concurrency() / 2);

    VadCtxGuard vctx{ whisper_vad_init_from_file_with_params(opts.vad_model_path.c_str(), cp) };
    if (!vctx.p) { err = "segment_mode: failed to load VAD model: " + opts.vad_model_path; return false; }

    whisper_vad_params vp = whisper_vad_default_params();
    vp.threshold              = opts.threshold;
    vp.min_speech_duration_ms = opts.min_speech_ms;
    vp.min_silence_duration_ms= opts.min_silence_ms;
    vp.max_speech_duration_s  = opts.max_speech_s;
    vp.speech_pad_ms          = opts.speech_pad_ms;

    VadSegsGuard segs{ whisper_vad_segments_from_samples(vctx.p, vp, pcm.data(), (int)pcm.size()) };
    if (!segs.p) { err = "segment_mode: VAD segmentation failed"; return false; }

    const int n = whisper_vad_segments_n_segments(segs.p);
    log("VAD found " + std::to_string(n) + " speech regions in "
        + std::to_string((int)total_sec) + "s of audio");

    // Export the detected speech regions (original timeline) if the caller wants them,
    // so the --debug report can correlate VAD regions vs cues in segment mode too.
    // NOTE: callers must gate the legacy Pass 2/3 block on the mode flag, not on
    // vad_regions being empty, since we now populate it here.
    if (base_topts.vad_regions) {
        base_topts.vad_regions->clear();
        for (int i = 0; i < n; ++i)
            base_topts.vad_regions->push_back({
                whisper_vad_segments_get_segment_t0(segs.p, i) * 0.01,
                whisper_vad_segments_get_segment_t1(segs.p, i) * 0.01 });
    }

    if (on_progress) on_progress(0);
    if (n <= 0) return true; // nothing to transcribe

    // --- Per-region isolated transcription (whisper VAD OFF, no concatenation) ---
    transcribe::Options topt = base_topts;
    topt.vad = false;
    topt.vad_model_path.clear();
    topt.vad_regions = nullptr;
    topt.on_segment  = nullptr;   // suppress per-slice callback spam
    topt.on_progress = nullptr;

    out.reserve(out.size() + (size_t)n);
    int emitted = 0;
    int dropped = 0;

    for (int i = 0; i < n; ++i) {
        double r0 = whisper_vad_segments_get_segment_t0(segs.p, i) * 0.01; // cs -> s
        double r1 = whisper_vad_segments_get_segment_t1(segs.p, i) * 0.01;
        if (r1 <= r0) continue;

        double s0 = std::max(0.0, r0 - opts.slice_pad_s);
        double s1 = std::min(total_sec, r1 + opts.slice_pad_s);

        size_t a = (size_t)(s0 * kSampleRate);
        size_t b = (size_t)(s1 * kSampleRate);
        if (b > pcm.size()) b = pcm.size();
        if (b <= a) continue;

        std::vector<float> slice(pcm.begin() + a, pcm.begin() + b);

        std::vector<srt::Segment> slice_segs;
        std::string serr;
        if (!session.transcribe(slice, topt, slice_segs, serr)) {
            log("region " + std::to_string(i) + " transcribe failed: " + serr);
            std::vector<float>().swap(slice);
            continue; // best-effort: skip this region, keep going
        }
        std::vector<float>().swap(slice); // free slice immediately

        for (auto& s : slice_segs) {
            // Precision pass: drop empty/punctuation-only cues and collapse
            // within-cue hallucinated phrase repetition. (Cross-cue duplicates are
            // handled downstream by srt::write's windowed dedup.)
            std::string cleaned = clean_cue_text(s.text);
            if (cleaned.empty()) { ++dropped; continue; }
            s.text = std::move(cleaned);
            // Slice-local time -> original timeline.
            s.t0 += s0;
            s.t1 += s0;
            // Keep cues within their padded slice so a hallucinated tail can't spill.
            s.t0 = std::max(s.t0, s0);
            s.t1 = std::min(s.t1, s1);
            if (s.t1 <= s.t0) continue;
            if (on_cue) on_cue(s);          // live transcript update
            out.push_back(std::move(s));
            ++emitted;
        }

        if (on_progress) on_progress((int)((long long)(i + 1) * 100 / n));
    }

    // Regions are already chronological, but padding overlap can reorder cue starts
    // slightly; sort defensively so srt::write indexes them correctly.
    std::stable_sort(out.begin(), out.end(),
                     [](const srt::Segment& x, const srt::Segment& y) { return x.t0 < y.t0; });

    log("segment mode emitted " + std::to_string(emitted) + " cues from "
        + std::to_string(n) + " regions (" + std::to_string(dropped)
        + " empty/repeat cues dropped)");
    return true;
}

} // namespace segment_mode
