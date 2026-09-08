#include "transcribe.h"

#include "whisper.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

namespace transcribe {

namespace {

// Route whisper/ggml's internal messages (model load, VAD info, warnings about
// failed timestamps / repetition, etc.) into our log file while still echoing to
// stderr. Without this the log said nothing when a run silently degraded.
void whisper_log_cb(ggml_log_level level, const char* text, void* /*ud*/) {
    if (!text || !*text) return;
    std::fputs(text, stderr); // preserve the console output users rely on
    if (level == GGML_LOG_LEVEL_CONT) return; // progress fragments: console only
    std::string s(text);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (s.empty()) return;
    const char* lvl = level >= GGML_LOG_LEVEL_ERROR ? "ERROR"
                    : level == GGML_LOG_LEVEL_WARN  ? "WARN" : "INFO";
    logging::logf(lvl, "whisper: %s", s.c_str());
}

void install_whisper_logging() {
    static std::once_flag once;
    std::call_once(once, [] { whisper_log_set(whisper_log_cb, nullptr); });
}

// Fired by whisper as new segments are decoded; forwards each to on_segment.
void on_new_segment(whisper_context* ctx, whisper_state* /*state*/,
                    int n_new, void* user_data) {
    const Options* o = static_cast<const Options*>(user_data);
    if (!o->on_segment) return;
    int n = whisper_full_n_segments(ctx);
    for (int i = n - n_new; i < n; ++i) {
        srt::Segment s;
        s.t0 = whisper_full_get_segment_t0(ctx, i) * 0.01;
        s.t1 = whisper_full_get_segment_t1(ctx, i) * 0.01;
        const char* txt = whisper_full_get_segment_text(ctx, i);
        s.text = txt ? txt : "";
        o->on_segment(s);
    }
}

void on_progress_cb(whisper_context* /*ctx*/, whisper_state* /*state*/,
                    int progress, void* user_data) {
    const Options* o = static_cast<const Options*>(user_data);
    if (o->on_progress) o->on_progress(progress);
}

} // namespace

bool run(const std::vector<float>& pcm, const Options& opts,
         std::vector<srt::Segment>& out, std::string& err) {
    out.clear();
    install_whisper_logging();

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = true;
    cparams.flash_attn = opts.flash_attn;
    cparams.gpu_device = 0;

    whisper_context* ctx =
        whisper_init_from_file_with_params(opts.model_path.c_str(), cparams);
    if (!ctx) { err = "failed to load model: " + opts.model_path; return false; }

    whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.print_progress   = opts.verbose;
    wp.print_realtime   = false;
    wp.print_timestamps = false;
    wp.print_special    = false;
    wp.translate        = opts.translate;
    // A language of "auto" makes whisper detect the language AND transcribe.
    // detect_language=true is a *detect-only* mode that returns with zero
    // segments - do not set it, or nothing gets transcribed.
    wp.language         = opts.language.c_str();
    wp.detect_language  = false;

    // English-only (.en) models cannot auto-detect; pin them to English.
    if (opts.language == "auto" && !whisper_is_multilingual(ctx)) {
        wp.language = "en";
    }
    wp.token_timestamps = opts.word_timestamps;
    // no_context = true: do NOT seed each 30s window with the previous window's
    // text. Conditioning on previous text (the default) is the main cause of
    // whisper's runaway repetition loops - once it emits a wrong repeated line it
    // feeds that back and gets stuck for the rest of the film (observed on
    // large-v3: a single line looped from 13:33 to the end, losing ~90 min). The
    // cost is slightly less cross-segment coherence; robustness on long movies is
    // the right trade. Temperature fallback (temperature_inc/entropy/logprob
    // tholds, on by default) is the second line of defense within a window.
    wp.no_context       = true;
    wp.suppress_nst     = true; // suppress non-speech tokens (curbs music hallucinations)

    int threads = opts.threads;
    if (threads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        threads = (int)std::max(1u, hc ? hc / 2 : 4u);
    }
    wp.n_threads = threads;

    if (opts.vad && !opts.vad_model_path.empty()) {
        wp.vad            = true;
        wp.vad_model_path = opts.vad_model_path.c_str();
        wp.vad_params     = whisper_vad_default_params();
        // Tuned for film mixes. The critical one is max_speech_duration_s: the
        // default is unbounded, so under continuous score VAD merges minutes of
        // audio into one "speech" region and whisper emits a single sparse cue
        // (wrong timing, most dialogue lost). Capping it forces fine segments.
        // Generous padding avoids clipping word edges.
        wp.vad_params.max_speech_duration_s  = 15.0f;
        wp.vad_params.speech_pad_ms          = 200;
    }

    if (opts.on_segment) {
        wp.new_segment_callback           = on_new_segment;
        wp.new_segment_callback_user_data = const_cast<Options*>(&opts);
    }
    if (opts.on_progress) {
        wp.progress_callback           = on_progress_cb;
        wp.progress_callback_user_data = const_cast<Options*>(&opts);
    }

    if (whisper_full(ctx, wp, pcm.data(), (int)pcm.size()) != 0) {
        err = "whisper_full failed";
        whisper_free(ctx);
        return false;
    }

    int n = whisper_full_n_segments(ctx);
    out.reserve((size_t)n);
    const whisper_token eot = whisper_token_eot(ctx);
    for (int i = 0; i < n; ++i) {
        srt::Segment s;
        s.t0 = whisper_full_get_segment_t0(ctx, i) * 0.01; // centiseconds -> s
        s.t1 = whisper_full_get_segment_t1(ctx, i) * 0.01;
        const char* txt = whisper_full_get_segment_text(ctx, i);
        s.text = txt ? txt : "";

        // Tighten start/end to the actual spoken words using per-token DTW times.
        // Whisper's segment-level t0 is back-dated across leading non-speech - plain
        // silence, or the loud residual vocal isolation leaves behind that VAD keeps
        // as "speech" - so cues appear seconds early. Cross-attention DTW locates
        // each word acoustically, so the first/last real token pins the true span.
        //
        // Use whisper_full_get_token_t0/t1 (NOT ...get_token_data().t0), because
        // those accessors remap the token time from the VAD-compressed timeline back
        // to the original one; the raw token_data times stay compressed and would
        // shift every cue. Requires word_timestamps (adds some cost); no-op off.
        if (opts.word_timestamps) {
            int nt = whisper_full_n_tokens(ctx, i);
            double first = -1.0, last = -1.0;
            for (int j = 0; j < nt; ++j) {
                whisper_token_data td = whisper_full_get_token_data(ctx, i, j);
                if (td.id >= eot || td.t0 < 0) continue; // skip specials / no DTW time
                double tt0 = whisper_full_get_token_t0(ctx, i, j) * 0.01; // VAD-remapped
                double tt1 = whisper_full_get_token_t1(ctx, i, j) * 0.01;
                if (first < 0.0) first = tt0;
                last = tt1;
            }
            if (first >= 0.0) s.t0 = first;
            if (last  >  s.t0) s.t1 = last;
        }
        out.push_back(std::move(s));
    }

    // Expose VAD speech regions (original timeline) for the debug report: they
    // let us see whether a cue's back-dated start falls inside a region VAD kept
    // as "speech" (e.g. loud isolation residual) vs a gap VAD correctly dropped.
    if (opts.vad_regions) {
        opts.vad_regions->clear();
        int nv = whisper_full_n_vad_segments(ctx);
        opts.vad_regions->reserve((size_t)(nv > 0 ? nv : 0));
        for (int i = 0; i < nv; ++i)
            opts.vad_regions->push_back({ whisper_full_get_vad_segment_t0(ctx, i) * 0.01,
                                          whisper_full_get_vad_segment_t1(ctx, i) * 0.01 });
    }

    // Diagnostic: report the raw (pre-dedup) segment span. If this ends far short
    // of the audio length, whisper stopped/stalled; if it reaches the end but the
    // final SRT is short, the de-dup collapsed a repetition loop (both look like
    // "it stopped early" to the user, but the cause and fix differ).
    double audio_sec = pcm.size() / 16000.0;
    if (!out.empty()) {
        logging::logf("INFO",
            "whisper: %d raw segments, span %.1fs..%.1fs of %.1fs audio (%.0f%% covered)",
            n, out.front().t0, out.back().t1, audio_sec,
            audio_sec > 0 ? out.back().t1 / audio_sec * 100.0 : 0.0);
        if (audio_sec > 0 && out.back().t1 < audio_sec - 30.0)
            logging::logf("WARN",
                "whisper: last segment ends %.1fs before end of audio - likely a "
                "decode stall or repetition loop; check for repeated text",
                audio_sec - out.back().t1);
    } else {
        logging::logf("WARN", "whisper: produced 0 segments for %.1fs audio", audio_sec);
    }

    whisper_free(ctx);
    return true;
}

} // namespace transcribe
