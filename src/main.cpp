#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "audio.h"
#include "transcribe.h"
#include "srt.h"
#include "models.h"
#include "separate.h"
#include "log.h"
#include "debug.h"
#include "segment_mode.h"

namespace {

const char* kDefaultModel = "large-v3-turbo-q8_0";

std::string format_duration_hms(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    long long total_s = (long long)(seconds + 0.5);
    long long h = total_s / 3600;
    long long m = (total_s % 3600) / 60;
    long long s = total_s % 60;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld", h, m, s);
    return std::string(buf);
}

void usage() {
    std::printf(
        "srt - generate an SRT subtitle file from any movie/audio file\n"
        "\n"
        "Usage:\n"
        "  srt <input> [options]\n"
        "\n"
        "Options:\n"
        "  -o, --output <path>      Output SRT (default: <input>.srt)\n"
        "  -m, --model <name|path>  Model name or file (default: %s)\n"
        "  -l, --language <code>    Source language or 'auto' (default: auto)\n"
        "      --translate          Translate to English\n"
        "      --audio-stream <n>   Audio stream index (default: best)\n"
        "      --duration <sec>     Only transcribe the first <sec> seconds\n"
        "      --no-flash-attn      Disable flash attention (on by default)\n"
        "      --center             Extract only the Front-Center channel (dialogue) before\n"
        "                           isolation (default OFF: isolate from full stereo downmix)\n"
        "      --vocal-model <name> Vocal model: Kim_Vocal_2 (default), etc.\n"
        "      --no-cache           Ignore any cached <stem>.vocals16k/.whisper16k.wav\n"
        "      --backend <name>     Compute backend: auto (default), cuda, vulkan, cpu.\n"
        "                           Only backends detected on this machine are usable.\n"
        "      --no-word-timestamps Disable DTW word timing (on by default; it\n"
        "                           snaps each cue to the actual spoken words)\n"
        "      --dump-audio [path]  Write the 16k mono audio whisper hears to a WAV\n"
        "                           (debug; default: next to the input file)\n"
        "      --time-offset <sec>  Shift every cue by <sec> (+later, -earlier)\n"
        "      --debug              Write <out>.debug.txt: VAD regions vs cues vs\n"
        "                           audio energy (diagnose early/late cue timing)\n"
        "      --reference <srt>    Ground-truth SRT to compare timing against in\n"
        "                           the --debug report (e.g. embedded subtitles)\n"
        "      --max-line-length <n> Wrap subtitles to <n> chars/line (default 42; 0=off)\n"
        "      --threads <n>        Worker threads (default: auto)\n"
        "      --models-dir <path>  Model store (default: %%LOCALAPPDATA%%\\SRTCreator\\models)\n"
        "      --download <name>    Download a model and exit\n"
        "      --verbose            Verbose progress\n"
        "  -h, --help               This help\n",
        kDefaultModel);
}

// Replace the input's extension with .srt (or append if none).
std::string default_output(const std::string& input) {
    size_t slash = input.find_last_of("/\\");
    size_t dot   = input.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return input + ".srt";
    return input.substr(0, dot) + ".srt";
}

// Fetch the value after a flag, advancing i. Returns false if missing.
bool take(int argc, char** argv, int& i, const char* flag, std::string& out) {
    if (i + 1 >= argc) { std::fprintf(stderr, "error: %s requires a value\n", flag); return false; }
    out = argv[++i];
    return true;
}

} // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    logging::init("cli");

    std::string input, output, model = kDefaultModel, language = "auto";
    std::string models_dir, download_name, threads_s, stream_s, duration_s, maxline_s;
    bool translate = false, flash = true, word_ts = true, verbose = false;
    bool center = false; // default OFF: isolate from the full stereo downmix (--center to
                         // extract only the Front-Center channel first)
    std::string vocal_model = "Kim_Vocal_2";
    int max_line_length = 42; // Netflix-style default; 0 disables wrapping
    bool dump_audio = false;            // write the 16k mono whisper input to a WAV
    std::string dump_audio_path;        // explicit dump path (else auto, next to exe)
    std::string offset_s;               // constant sync shift (seconds; +later, -earlier)
    bool debug = false;                 // write a timing diagnostic report
    std::string reference_path;         // optional reference SRT for the debug report
    bool use_cache = true;              // reuse an existing <stem>.vocals16k/.whisper16k.wav
    std::string backend_name;           // --backend auto|cuda|vulkan|cpu (empty = auto)

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-h" || a == "--help")   { usage(); return 0; }
        else if (a == "-o" || a == "--output") { if (!take(argc, argv, i, "--output", output)) return 2; }
        else if (a == "-m" || a == "--model")  { if (!take(argc, argv, i, "--model", model)) return 2; }
        else if (a == "-l" || a == "--language") { if (!take(argc, argv, i, "--language", language)) return 2; }
        else if (a == "--translate")       translate = true;
        else if (a == "--audio-stream")    { if (!take(argc, argv, i, "--audio-stream", stream_s)) return 2; }
        else if (a == "--duration")        { if (!take(argc, argv, i, "--duration", duration_s)) return 2; }
        else if (a == "--max-line-length") { if (!take(argc, argv, i, "--max-line-length", maxline_s)) return 2; }
        else if (a == "--no-flash-attn")   flash = false;
        else if (a == "--flash-attn")      flash = true;
        else if (a == "--no-vad" || a == "--vad") { /* VAD is always enabled */ }
        else if (a == "--no-center")       center = false;
        else if (a == "--center")          center = true;
        else if (a == "--isolate-vocals" || a == "--vocals") { /* Vocal isolation is always enabled */ }
        else if (a == "--vocal-model")     { if (!take(argc, argv, i, "--vocal-model", vocal_model)) return 2; }
        else if (a == "--no-cache")        use_cache = false;
        else if (a == "--backend")         { if (!take(argc, argv, i, "--backend", backend_name)) return 2; }
        else if (a == "--word-timestamps") word_ts = true;
        else if (a == "--no-word-timestamps") word_ts = false;
        else if (a == "--dump-audio")      { dump_audio = true; if (i + 1 < argc && argv[i+1][0] != '-') dump_audio_path = argv[++i]; }
        else if (a == "--time-offset")     { if (!take(argc, argv, i, "--time-offset", offset_s)) return 2; }
        else if (a == "--debug")           debug = true;
        else if (a == "--reference")       { if (!take(argc, argv, i, "--reference", reference_path)) return 2; }
        else if (a == "--threads")         { if (!take(argc, argv, i, "--threads", threads_s)) return 2; }
        else if (a == "--models-dir")      { if (!take(argc, argv, i, "--models-dir", models_dir)) return 2; }
        else if (a == "--download")        { if (!take(argc, argv, i, "--download", download_name)) return 2; }
        else if (a == "--verbose")         verbose = true;
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "error: unknown option %s\n", a.c_str()); return 2; }
        else if (input.empty())            input = a;
        else { std::fprintf(stderr, "error: unexpected argument %s\n", a.c_str()); return 2; }
    }

    if (models_dir.empty()) models_dir = models::default_models_dir();
    std::string err;

    // --download <name>: fetch and exit.
    if (!download_name.empty()) {
        std::string path;
        if (!models::download(download_name, models_dir, path, err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        std::printf("downloaded: %s\n", path.c_str());
        return 0;
    }

    if (input.empty()) { usage(); return 2; }
    if (output.empty()) output = default_output(input);

    int threads = threads_s.empty() ? 0 : std::atoi(threads_s.c_str());
    int stream  = stream_s.empty()  ? -1 : std::atoi(stream_s.c_str());
    if (!maxline_s.empty()) max_line_length = std::atoi(maxline_s.c_str());

    // 1) Resolve model (download if needed).
    std::string model_path;
    if (!models::resolve(model, models_dir, /*allow_download=*/true, model_path, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }

    // 2) VAD model (always enabled).
    std::string vad_path;
    if (!models::ensure_vad(models_dir, /*allow_download=*/true, vad_path, err)) {
        std::fprintf(stderr, "error: VAD model required: %s\n", err.c_str());
        return 1;
    }

    logging::logf("INFO", "input=%s model=%s lang=%s translate=%d center=%d wrap=%d",
                  input.c_str(), model_path.c_str(), language.c_str(),
                  (int)translate, (int)center, max_line_length);

    // 3) Decode audio (44.1k stereo) and isolate vocals -> 16 kHz mono f32.
    auto t_job_start = std::chrono::steady_clock::now();
    double dur_phase1 = 0.0; // Audio extraction
    double dur_phase2 = 0.0; // Voice isolation
    double dur_phase3 = 0.0; // Speech transcription

    double maxsec = duration_s.empty() ? 0.0 : std::atof(duration_s.c_str());
    std::vector<float> pcm;

    // Cache: if a 16k mono whisper input already sits next to the movie (a prior
    // run's dump, or the GUI's), load it directly and skip decode + isolation. This
    // is a big time saver when iterating on transcription/timeline logic - and it
    // avoids re-running the GPU isolation entirely. Prefer the isolated vocal stem.
    std::string cached_wav;
    if (use_cache) {
        size_t cdot = input.find_last_of('.');
        std::string cbase = (cdot == std::string::npos) ? input : input.substr(0, cdot);
        for (const char* suf : { ".vocals16k.wav", ".whisper16k.wav" }) {
            std::string cand = cbase + suf;
            FILE* cf = std::fopen(cand.c_str(), "rb");
            if (cf) { std::fclose(cf); cached_wav = cand; break; }
        }
    }

    if (!cached_wav.empty()) {
        std::fprintf(stderr, "[srt] using cached audio (skipping decode + isolation): %s\n", cached_wav.c_str());
        logging::logf("INFO", "using cached whisper input %s (skip decode/isolation)", cached_wav.c_str());
        auto t_c0 = std::chrono::steady_clock::now();
        audio::DecodeOptions dc;
        dc.sample_rate = 16000; dc.channels = 1;
        dc.stream_index = -1; dc.max_seconds = maxsec; dc.center_channel_only = false;
        if (!audio::decode(cached_wav, dc, pcm, err)) {
            logging::error("cached audio load failed: " + err);
            std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
        }
        dur_phase1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_c0).count();
        logging::logf("INFO", "Phase 1 (Cached audio load) completed in %s (%.2fs)",
                      format_duration_hms(dur_phase1).c_str(), dur_phase1);
        std::fprintf(stderr, "[timing] Phase 1 (Cached audio load): %s\n", format_duration_hms(dur_phase1).c_str());
        std::fprintf(stderr, "[srt] cached audio ready: %.1f min\n", pcm.size() / 16000.0 / 60.0);
    } else {
    std::fprintf(stderr, "[srt] decoding (44.1k stereo for separation): %s\n", input.c_str());
    logging::info("decoding for separation");
    audio::DecodeOptions d;
    d.sample_rate = 44100; d.channels = 2;
    d.stream_index = stream; d.max_seconds = maxsec; d.center_channel_only = center;
    std::vector<float> mix;
    auto t_p1_start = std::chrono::steady_clock::now();
    if (!audio::decode(input, d, mix, err)) {
        logging::error("decode failed: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
    }
    auto t_p1_end = std::chrono::steady_clock::now();
    dur_phase1 = std::chrono::duration<double>(t_p1_end - t_p1_start).count();
    logging::logf("INFO", "Phase 1 (Audio extraction) completed in %s (%.2fs)",
                  format_duration_hms(dur_phase1).c_str(), dur_phase1);
    std::fprintf(stderr, "[timing] Phase 1 (Audio extraction): %s\n", format_duration_hms(dur_phase1).c_str());

    std::string mpath; separate::Params sp;
    if (!separate::ensure_model(vocal_model, models_dir, true, mpath, sp, err)) {
        logging::error("vocal model: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
    }
    std::fprintf(stderr, "[srt] isolating vocals (%s) ...\n", vocal_model.c_str());
    logging::logf("INFO", "isolating vocals with %s", vocal_model.c_str());
    std::vector<float> vocals;
    auto t_p2_start = std::chrono::steady_clock::now();
    if (!separate::isolate_vocals(mix, mpath, sp, vocals, nullptr, err)) {
        logging::error("separation failed: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
    }
    std::vector<float>().swap(mix); // free the 44.1k stereo mix before resampling
    if (!audio::resample_to_mono(vocals, 44100, 1, 16000, pcm, err)) {
        std::vector<float>().swap(vocals);
        logging::error("resample failed: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
    }
    std::vector<float>().swap(vocals); // IMMEDIATELY free 44.1k mono vocals (~1.6GB)
    auto t_p2_end = std::chrono::steady_clock::now();
    dur_phase2 = std::chrono::duration<double>(t_p2_end - t_p2_start).count();
    logging::logf("INFO", "Phase 2 (Voice isolation) completed in %s (%.2fs)",
                  format_duration_hms(dur_phase2).c_str(), dur_phase2);
    std::fprintf(stderr, "[timing] Phase 2 (Voice isolation): %s\n", format_duration_hms(dur_phase2).c_str());

    std::fprintf(stderr, "[srt] isolated audio ready: %.1f min\n", pcm.size() / 16000.0 / 60.0);
    logging::logf("INFO", "isolated audio ready %.1f min (%zu samples)", pcm.size() / 16000.0 / 60.0, pcm.size());
    }

    // Optional: dump audio (vocals)
    if (dump_audio) {
        std::string wav = dump_audio_path;
        if (wav.empty()) {
            size_t dot = input.find_last_of('.');
            std::string base = (dot == std::string::npos) ? input : input.substr(0, dot);
            wav = base + ".vocals16k.wav";
        }
        std::string werr;
        if (audio::write_wav(wav, pcm, 16000, 1, werr)) {
            std::fprintf(stderr, "[srt] wrote debug audio: %s\n", wav.c_str());
            logging::logf("INFO", "dumped whisper input audio to %s", wav.c_str());
        } else {
            std::fprintf(stderr, "warning: could not write debug audio (%s)\n", werr.c_str());
            logging::error("audio dump failed: " + werr);
        }
    }

    // 4) Transcribe.
    std::fprintf(stderr, "[srt] transcribing with %s ...\n", model_path.c_str());
    transcribe::Options topts;
    topts.model_path      = model_path;
    topts.language        = language;
    topts.translate       = translate;
    topts.flash_attn      = flash;
    topts.vad             = true;
    topts.vad_model_path  = vad_path;
    topts.threads         = threads;
    topts.word_timestamps = word_ts;
    topts.verbose         = verbose;
    std::vector<transcribe::VadRegion> vad_regions;
    topts.vad_regions     = &vad_regions;

    // Resolve --backend <name> to a device index (empty = auto). Names are matched
    // against the backends actually detected on this machine.
    if (!backend_name.empty() && backend_name != "auto") {
        auto devs = transcribe::available_devices();
        std::string want = backend_name;
        for (auto& c : want) c = (char)std::tolower((unsigned char)c);
        int found = -1;
        for (size_t di = 0; di < devs.size(); ++di) {
            std::string nm = devs[di].name;
            for (auto& c : nm) c = (char)std::tolower((unsigned char)c);
            bool is_cpu = !devs[di].is_gpu;
            if ((want == "cpu" && is_cpu) ||
                (want == "cuda" && nm.rfind("cuda", 0) == 0) ||
                (want == "vulkan" && nm.rfind("vulkan", 0) == 0)) { found = (int)di; break; }
        }
        if (found < 0) {
            std::string avail;
            for (auto& d : devs) { if (!avail.empty()) avail += ", "; avail += d.name; }
            std::fprintf(stderr, "error: backend '%s' not available on this machine. Detected: %s\n",
                         backend_name.c_str(), avail.c_str());
            return 2;
        }
        topts.device_index = found;
    }

    logging::info("transcribing");
    std::vector<srt::Segment> segments;
    transcribe::Session session;
    auto t_p3_start = std::chrono::steady_clock::now();
    if (!session.init(topts, err)) {
        logging::error("whisper init failed: " + err);
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    // Pipeline: standalone Silero VAD partitions the track into speech regions and
    // each is transcribed in isolation (no whisper-VAD concatenation, no seam), so
    // cue timing is just region_start + local time. See segment_mode.{h,cpp}.
    std::fprintf(stderr, "[srt] segment mode: VAD-segmented per-region transcription\n");
    segment_mode::Options sopts;
    sopts.vad_model_path = vad_path;
    if (!segment_mode::run(pcm, session, topts, sopts, segments, err,
            [](const std::string& m){ std::fprintf(stderr, "  [segment] %s\n", m.c_str()); })) {
        logging::error("segment-mode failed: " + err);
        session.close();
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    auto t_p3_end = std::chrono::steady_clock::now();
    dur_phase3 = std::chrono::duration<double>(t_p3_end - t_p3_start).count();
    logging::logf("INFO", "Phase 3 (Transcription) completed in %s (%.2fs)",
                  format_duration_hms(dur_phase3).c_str(), dur_phase3);
    std::fprintf(stderr, "[timing] Phase 3 (Transcription): %s\n", format_duration_hms(dur_phase3).c_str());
    logging::logf("INFO", "transcribed %zu segments", segments.size());

    session.close(); // explicitly release Whisper model and reset CUDA device memory

    // Optional constant sync shift.
    double time_offset = offset_s.empty() ? 0.0 : std::atof(offset_s.c_str());
    if (time_offset != 0.0) {
        for (auto& s : segments) {
            s.t0 += time_offset; if (s.t0 < 0.0) s.t0 = 0.0;
            s.t1 += time_offset; if (s.t1 < 0.0) s.t1 = 0.0;
        }
        logging::logf("INFO", "applied time offset %.3f s", time_offset);
    }

    // 5) Write SRT.
    if (!srt::write(segments, output, max_line_length, err)) {
        logging::error("write failed: " + err);
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    logging::logf("INFO", "wrote %zu segments to %s", segments.size(), output.c_str());
    std::printf("wrote %zu segments to %s\n", segments.size(), output.c_str());

    auto t_job_end = std::chrono::steady_clock::now();
    double dur_total = std::chrono::duration<double>(t_job_end - t_job_start).count();

    logging::logf("INFO", "Timing Summary: 1.Audio extraction=%s (%.2fs) | 2.Voice isolation=%s (%.2fs) | 3.Transcription=%s (%.2fs) | Total=%s (%.2fs)",
                  format_duration_hms(dur_phase1).c_str(), dur_phase1,
                  format_duration_hms(dur_phase2).c_str(), dur_phase2,
                  format_duration_hms(dur_phase3).c_str(), dur_phase3,
                  format_duration_hms(dur_total).c_str(), dur_total);

    std::fprintf(stderr,
                 "\n----------------------------------------\n"
                 "Timing Summary:\n"
                 "  1. Audio extraction:      %s\n"
                 "  2. Voice isolation:       %s\n"
                 "  3. Speech transcription:  %s\n"
                 "  Total time taken:         %s\n"
                 "  Cues written:             %zu\n"
                 "----------------------------------------\n\n",
                 format_duration_hms(dur_phase1).c_str(),
                 format_duration_hms(dur_phase2).c_str(),
                 format_duration_hms(dur_phase3).c_str(),
                 format_duration_hms(dur_total).c_str(),
                 segments.size());

    // 6) Optional timing diagnostics (--debug): correlate VAD regions, whisper
    // cues, actual audio energy, and an optional reference SRT.
    if (debug) {
        std::vector<srt::Segment> ref;
        const std::vector<srt::Segment>* refp = nullptr;
        if (!reference_path.empty()) {
            std::string rerr;
            if (srt::read(reference_path, ref, rerr) && !ref.empty()) refp = &ref;
            else std::fprintf(stderr, "warning: could not use reference %s (%s)\n",
                              reference_path.c_str(), rerr.c_str());
        }
        debugreport::Info di;
        di.model = model; di.isolate = true; di.center = center; di.vad = true; di.word_ts = word_ts;
        std::string dbg_path = output + ".debug.txt";
        std::string derr;
        if (debugreport::write(dbg_path, pcm, segments, vad_regions, refp, di, derr)) {
            std::fprintf(stderr, "[srt] wrote debug report: %s\n", dbg_path.c_str());
            logging::logf("INFO", "wrote debug report to %s", dbg_path.c_str());
        } else {
            std::fprintf(stderr, "warning: debug report failed (%s)\n", derr.c_str());
        }
    }

    std::vector<float>().swap(pcm);
    std::vector<srt::Segment>().swap(segments);
    std::vector<transcribe::VadRegion>().swap(vad_regions);

    return 0;
}
