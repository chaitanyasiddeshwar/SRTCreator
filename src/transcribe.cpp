#include "transcribe.h"

#include "whisper.h"
#include "ggml-backend.h"   // ggml_backend_load_all + device enumeration (GGML_BACKEND_DL)
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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

// In a GGML_BACKEND_DL build the backends live in separate DLLs next to the exe
// (ggml-cuda.dll / ggml-vulkan.dll / ggml-cpu-*.dll) and NONE are registered until
// we ask ggml to load them. Must run before any whisper/ggml device use, once.
void ensure_backends_loaded() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_backend_load_all();
        logging::logf("INFO", "ggml: loaded %zu compute device(s) via dynamic backends",
                      ggml_backend_dev_count());
    });
}

const char* dev_type_str(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "iGPU";
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default:                             return "other";
    }
}

bool dev_is_gpu(enum ggml_backend_dev_type t) {
    return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
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

static void reset_gpu_memory() {
#ifdef _WIN32
    typedef int (*cuda_fn)();
    const char* dlls[] = { "cudart64_13.dll", "cudart64_12.dll", "cudart64_11.dll", "cudart64_10.dll" };
    for (const char* dll : dlls) {
        HMODULE h = GetModuleHandleA(dll);
        if (!h) h = LoadLibraryA(dll);
        if (h) {
            auto pReset = (cuda_fn)GetProcAddress(h, "cudaDeviceReset");
            if (pReset) {
                pReset();
                logging::info("cuda: device reset called successfully");
            }
            break;
        }
    }
#endif
}

} // namespace

Session::Session() : ctx_(nullptr) {}

Session::~Session() {
    close();
}

Session::Session(Session&& other) noexcept : ctx_(other.ctx_) {
    other.ctx_ = nullptr;
}

Session& Session::operator=(Session&& other) noexcept {
    if (this != &other) {
        close();
        ctx_ = other.ctx_;
        other.ctx_ = nullptr;
    }
    return *this;
}

bool Session::is_valid() const {
    return ctx_ != nullptr;
}

bool Session::init(const Options& opts, std::string& err) {
    close();
    install_whisper_logging();
    ensure_backends_loaded();   // register the plugin backend DLLs before device use

    // available_devices() returns GPUs first (ggml order) then CPU, so a GPU's list
    // index equals whisper's gpu_device (the index among GPU-type devices), and the
    // CPU entry maps to use_gpu=false. Only backends whose DLL actually loaded appear.
    std::vector<DeviceInfo> devs = available_devices();
    std::string report;
    for (const auto& d : devs) {
        if (!report.empty()) report += ", ";
        report += d.name + " (" + (d.is_gpu ? "GPU" : "CPU") + ")";
    }

    // Resolve the requested device. -1 (auto) = the best GPU if any, else CPU.
    bool has_gpu = !devs.empty() && devs.front().is_gpu;
    int  sel     = opts.device_index;
    bool use_gpu;
    int  gpu_dev = 0;
    std::string chosen;
    if (sel >= 0 && sel < (int)devs.size()) {
        use_gpu = devs[sel].is_gpu;
        gpu_dev = use_gpu ? sel : 0;    // GPUs are first, so list index == gpu_device
        chosen  = devs[sel].name + (use_gpu ? " (forced GPU)" : " (forced CPU)");
    } else {
        use_gpu = has_gpu;
        chosen  = has_gpu ? (devs.front().name + " (auto)") : "CPU (auto, no GPU backend found)";
    }
    logging::logf("INFO", "backends: %s -> using %s", report.c_str(), chosen.c_str());
    std::fprintf(stderr, "[srt] compute backends: %s -> %s\n", report.c_str(), chosen.c_str());
    if (!has_gpu && sel < 0) {
        std::fprintf(stderr, "[srt] no GPU backend found - running on CPU (slower). "
                             "Drop ggml-cuda.dll or ggml-vulkan.dll next to the exe for GPU.\n");
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = use_gpu;
    cparams.flash_attn = opts.flash_attn;
    cparams.gpu_device = gpu_dev;

    ctx_ = whisper_init_from_file_with_params(opts.model_path.c_str(), cparams);
    if (!ctx_) {
        err = "failed to load model: " + opts.model_path;
        return false;
    }
    return true;
}

void Session::close() {
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
        reset_gpu_memory();
    }
}

bool Session::transcribe(const std::vector<float>& pcm, const Options& opts,
                         std::vector<srt::Segment>& out, std::string& err) {
    out.clear();
    if (!ctx_) {
        err = "whisper session not initialized";
        return false;
    }
    if (pcm.empty()) return true;

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
    if (opts.language == "auto" && !whisper_is_multilingual(ctx_)) {
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

    if (whisper_full(ctx_, wp, pcm.data(), (int)pcm.size()) != 0) {
        err = "whisper_full failed";
        return false;
    }

    int n = whisper_full_n_segments(ctx_);
    out.reserve((size_t)n);
    const whisper_token eot = whisper_token_eot(ctx_);
    for (int i = 0; i < n; ++i) {
        srt::Segment s;
        s.t0 = whisper_full_get_segment_t0(ctx_, i) * 0.01; // centiseconds -> s
        s.t1 = whisper_full_get_segment_t1(ctx_, i) * 0.01;
        const char* txt = whisper_full_get_segment_text(ctx_, i);
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
            int nt = whisper_full_n_tokens(ctx_, i);
            double first = -1.0, last = -1.0;
            double first_t1 = -1.0;
            int first_len = 0;
            double last_t0 = -1.0;
            int last_len = 0;
            for (int j = 0; j < nt; ++j) {
                whisper_token_data td = whisper_full_get_token_data(ctx_, i, j);
                if (td.id >= eot || td.t0 < 0) continue; // skip specials / no DTW time
                double tt0 = whisper_full_get_token_t0(ctx_, i, j) * 0.01; // VAD-remapped
                double tt1 = whisper_full_get_token_t1(ctx_, i, j) * 0.01;
                const char* tstr = whisper_token_to_str(ctx_, td.id);
                int tlen = 0;
                if (tstr) {
                    while (*tstr == ' ' || *tstr == '\t' || *tstr == '\n') ++tstr;
                    tlen = (int)std::strlen(tstr);
                }
                if (first < 0.0) {
                    first = tt0;
                    first_t1 = tt1;
                    first_len = tlen;
                }
                last = tt1;
                last_t0 = tt0;
                last_len = tlen;
            }
            if (first >= 0.0) {
                // If the first token duration is suspiciously long, Whisper lacked a
                // timestamp token before it and back-dated token 0 across leading silence.
                double max_dur = std::max(0.35, 0.06 * first_len + 0.25);
                if (first_t1 > first && (first_t1 - first) > max_dur) {
                    first = std::max(first, first_t1 - max_dur);
                }
                s.t0 = first;
            }
            if (last > s.t0) {
                // If the last token duration is suspiciously long, Whisper lacked a
                // trailing timestamp and forward-dated token N across trailing silence.
                double max_last_dur = std::max(0.40, 0.08 * last_len + 0.35);
                if (last_t0 >= 0.0 && (last - last_t0) > max_last_dur) {
                    last = last_t0 + max_last_dur;
                }
                s.t1 = last;
            }
        }
        out.push_back(std::move(s));
    }

    // Expose VAD speech regions (original timeline)
    std::vector<transcribe::VadRegion> vad_segs;
    int nv = whisper_full_n_vad_segments(ctx_);
    vad_segs.reserve((size_t)(nv > 0 ? nv : 0));
    for (int i = 0; i < nv; ++i)
        vad_segs.push_back({ whisper_full_get_vad_segment_t0(ctx_, i) * 0.01,
                             whisper_full_get_vad_segment_t1(ctx_, i) * 0.01 });
    if (opts.vad_regions) {
        *opts.vad_regions = vad_segs;
    }

    // VAD speech region containment:
    // If a short subtitle cue spans across a removed silence gap between two VAD regions,
    // Whisper forward-dated its end (or back-dated its start) across the cut-out gap.
    if (!vad_segs.empty()) {
        for (auto& s : out) {
            if (s.t0 >= s.t1) continue;
            int v_start = -1;
            for (int v = 0; v < (int)vad_segs.size(); ++v) {
                if (s.t0 >= vad_segs[v].t0 && s.t0 <= vad_segs[v].t1) {
                    v_start = v;
                    break;
                }
            }
            if (v_start >= 0 && v_start + 1 < (int)vad_segs.size()) {
                double gap = vad_segs[v_start + 1].t0 - vad_segs[v_start].t1;
                if (gap > 0.80 && s.t1 > vad_segs[v_start].t1 + 0.20) {
                    double vad_dur = vad_segs[v_start].t1 - s.t0;
                    if (s.text.size() < 60 || s.text.size() / std::max(0.5, vad_dur) < 25.0) {
                        s.t1 = std::min(s.t1, vad_segs[v_start].t1 + 0.150);
                    }
                }
            }
        }
    }

    // Acoustic onset and offset tightening:
    // When whisper begins a cue during leading silence or vocal isolation residual,
    // s.t0 can precede spoken dialogue. If audio energy at s.t0 is quiet (< -38 dBFS),
    // scan forward in pcm for sustained voiced energy and snap s.t0 forward.
    // Similarly, if a cue's end s.t1 extends into trailing silence (> 1.8s duration),
    // scan backward to find where the speech finished so subtitles don't hang frozen on screen.
    if (!pcm.empty() && !out.empty()) {
        const int bin_n = 320; // 20ms @ 16kHz
        const size_t n_bins = pcm.size() / bin_n;
        std::vector<float> db_bins(n_bins, -90.0f);
        for (size_t b = 0; b < n_bins; ++b) {
            size_t offset = b * bin_n;
            double sum = 0.0;
            for (int k = 0; k < bin_n; ++k) {
                float v = pcm[offset + k];
                sum += (double)v * v;
            }
            double rms = std::sqrt(sum / bin_n);
            db_bins[b] = (rms > 1e-9) ? (float)(20.0 * std::log10(rms)) : -90.0f;
        }

        for (auto& s : out) {
            if (s.t0 >= s.t1) continue;
            // Onset tightening:
            int b_start = (int)(s.t0 / 0.020);
            if (b_start >= 0 && b_start < (int)db_bins.size()) {
                if (db_bins[b_start] < -38.0f) {
                    int b_end = std::min((int)db_bins.size(), (int)(std::min(s.t1 - 0.10, s.t0 + 2.5) / 0.020));
                    for (int b = b_start; b + 1 < b_end; ++b) {
                        if (db_bins[b] >= -36.0f && db_bins[b + 1] >= -38.0f) {
                            double onset = b * 0.020;
                            double tightened = std::max(s.t0, onset - 0.050);
                            if (tightened < s.t1 - 0.050) {
                                s.t0 = tightened;
                            }
                            break;
                        }
                    }
                }
            }
            // Offset tightening:
            int b_end = (int)(s.t1 / 0.020);
            if (s.t1 - s.t0 > 1.8 && b_end >= 0 && b_end < (int)db_bins.size()) {
                if (db_bins[b_end] < -38.0f) {
                    int b_floor = std::max(0, (int)(s.t0 / 0.020));
                    for (int b = b_end; b > b_floor; --b) {
                        if (db_bins[b] >= -36.0f) {
                            double offset_t = b * 0.020 + 0.150;
                            double min_read_dur = std::min(2.5, 0.05 * s.text.size() + 0.80);
                            double tightened_end = std::max(s.t0 + min_read_dur, offset_t);
                            if (tightened_end < s.t1) {
                                s.t1 = tightened_end;
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    // Ensure comfortable reading duration:
    // A subtitle that flashes on screen for only 200-400ms is difficult to read.
    // Extend end time up to standard reading duration (at least 1.0s, scaled by length),
    // provided it does not collide with the next subtitle cue.
    double audio_sec = pcm.size() / 16000.0;
    for (size_t k = 0; k < out.size(); ++k) {
        if (out[k].t0 >= out[k].t1) continue;
        double min_read = std::min(2.5, std::max(1.0, 0.04 * (double)out[k].text.size() + 0.60));
        double target_end = std::max(out[k].t1, out[k].t0 + min_read);
        if (k + 1 < out.size() && out[k + 1].t0 > out[k].t0) {
            target_end = std::min(target_end, out[k + 1].t0 - 0.050);
        }
        if (audio_sec > 0.0) {
            target_end = std::min(target_end, audio_sec);
        }
        if (target_end > out[k].t0 + 0.200) {
            out[k].t1 = target_end;
        }
    }

    // Diagnostic: report the raw (pre-dedup) segment span.
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

    return true;
}

bool run(const std::vector<float>& pcm, const Options& opts,
         std::vector<srt::Segment>& out, std::string& err) {
    Session session;
    if (!session.init(opts, err)) return false;
    return session.transcribe(pcm, opts, out, err);
}

std::vector<DeviceInfo> available_devices() {
    ensure_backends_loaded();
    std::vector<DeviceInfo> gpus, cpus;
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
        DeviceInfo di;
        const char* name = ggml_backend_dev_name(d);
        const char* desc = ggml_backend_dev_description(d);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
        const char* rn = reg ? ggml_backend_reg_name(reg) : nullptr;
        di.name        = name ? name : "?";
        di.description = desc ? desc : "";
        di.backend     = rn ? rn : di.name;
        di.is_gpu      = dev_is_gpu(t);
        (di.is_gpu ? gpus : cpus).push_back(std::move(di));
    }
    // GPUs first (that is the whisper selection order too), then CPU.
    gpus.insert(gpus.end(), cpus.begin(), cpus.end());
    return gpus;
}

} // namespace transcribe
