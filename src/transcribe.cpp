#include "transcribe.h"

#include "whisper.h"

#include <algorithm>
#include <thread>

namespace transcribe {

namespace {

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
    wp.no_context       = false;
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
    for (int i = 0; i < n; ++i) {
        srt::Segment s;
        s.t0 = whisper_full_get_segment_t0(ctx, i) * 0.01; // centiseconds -> s
        s.t1 = whisper_full_get_segment_t1(ctx, i) * 0.01;
        const char* txt = whisper_full_get_segment_text(ctx, i);
        s.text = txt ? txt : "";
        out.push_back(std::move(s));
    }

    whisper_free(ctx);
    return true;
}

} // namespace transcribe
