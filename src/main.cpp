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

namespace {

const char* kDefaultModel = "large-v3-turbo-q8_0";

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
        "      --no-vad             Disable Silero VAD (on by default)\n"
        "      --no-center          Don't isolate the center channel (dialogue)\n"
        "      --isolate-vocals     Remove music/effects with a separation model first\n"
        "      --vocal-model <name> Vocal model: Kim_Vocal_2 (default), etc.\n"
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
    bool translate = false, flash = true, vad = true, word_ts = true, verbose = false;
    bool center = true; // isolate Front-Center (dialogue) for multichannel sources
    bool isolate = false;               // vocal isolation (music removal)
    std::string vocal_model = "Kim_Vocal_2";
    int max_line_length = 42; // Netflix-style default; 0 disables wrapping
    bool dump_audio = false;            // write the 16k mono whisper input to a WAV
    std::string dump_audio_path;        // explicit dump path (else auto, next to exe)
    std::string offset_s;               // constant sync shift (seconds; +later, -earlier)
    bool debug = false;                 // write a timing diagnostic report
    std::string reference_path;         // optional reference SRT for the debug report

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
        else if (a == "--no-vad")          vad = false;
        else if (a == "--vad")             vad = true;
        else if (a == "--no-center")       center = false;
        else if (a == "--center")          center = true;
        else if (a == "--isolate-vocals" || a == "--vocals") isolate = true;
        else if (a == "--vocal-model")     { if (!take(argc, argv, i, "--vocal-model", vocal_model)) return 2; }
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

    // Note: VAD stays ON even when isolating. It was once disabled here (music is
    // gone after separation), but without VAD whisper back-dates segment starts
    // across silence gaps (cues appear seconds early) and falls into repetition
    // loops on isolated-stem artifacts. Running Silero VAD on the isolated stem
    // fixes both - it trims the leading silence so timing tracks the real onset.

    // 2) VAD model (optional; disable gracefully if unavailable).
    std::string vad_path;
    if (vad) {
        if (!models::ensure_vad(models_dir, /*allow_download=*/true, vad_path, err)) {
            std::fprintf(stderr, "warning: VAD disabled (%s)\n", err.c_str());
            vad = false;
        }
    }

    logging::logf("INFO", "input=%s model=%s lang=%s translate=%d vad=%d wrap=%d",
                  input.c_str(), model_path.c_str(), language.c_str(),
                  (int)translate, (int)vad, max_line_length);

    // 3) Decode audio (and optionally isolate vocals) -> 16 kHz mono f32.
    double maxsec = duration_s.empty() ? 0.0 : std::atof(duration_s.c_str());
    std::vector<float> pcm;
    if (isolate) {
        std::fprintf(stderr, "[srt] decoding (44.1k stereo for separation): %s\n", input.c_str());
        logging::info("decoding for separation");
        audio::DecodeOptions d;
        d.sample_rate = 44100; d.channels = 2;
        d.stream_index = stream; d.max_seconds = maxsec; d.center_channel_only = center;
        std::vector<float> mix;
        if (!audio::decode(input, d, mix, err)) {
            logging::error("decode failed: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
        }
        std::string mpath; separate::Params sp;
        if (!separate::ensure_model(vocal_model, models_dir, true, mpath, sp, err)) {
            logging::error("vocal model: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
        }
        std::fprintf(stderr, "[srt] isolating vocals (%s) ...\n", vocal_model.c_str());
        logging::logf("INFO", "isolating vocals with %s", vocal_model.c_str());
        std::vector<float> vocals;
        if (!separate::isolate_vocals(mix, mpath, sp, vocals, nullptr, err)) {
            logging::error("separation failed: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
        }
        std::vector<float>().swap(mix); // free the 44.1k stereo mix before resampling
        if (!audio::resample_to_mono(vocals, 44100, 1, 16000, pcm, err)) {
            logging::error("resample failed: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
        }
    } else {
        std::fprintf(stderr, "[srt] decoding audio: %s\n", input.c_str());
        logging::info("decoding audio");
        audio::DecodeOptions dopts;
        dopts.stream_index = stream; dopts.max_seconds = maxsec; dopts.center_channel_only = center;
        if (!audio::decode(input, dopts, pcm, err)) {
            logging::error("decode failed: " + err); std::fprintf(stderr, "error: %s\n", err.c_str()); return 1;
        }
    }
    std::fprintf(stderr, "[srt] audio ready: %.1f min\n", pcm.size() / 16000.0 / 60.0);
    logging::logf("INFO", "audio ready %.1f min (%zu samples)", pcm.size() / 16000.0 / 60.0, pcm.size());

    // Optional: dump the exact 16 kHz mono audio whisper will hear (the isolated
    // vocal stem when --isolate-vocals). ONLY when --dump-audio is given - we never
    // need this file internally, so isolating alone must not leave one behind.
    // Default location is next to the input.
    if (dump_audio) {
        std::string wav = dump_audio_path;
        if (wav.empty()) {
            size_t dot = input.find_last_of('.');
            std::string base = (dot == std::string::npos) ? input : input.substr(0, dot);
            wav = base + (isolate ? ".vocals16k.wav" : ".whisper16k.wav");
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
    topts.vad             = vad;
    topts.vad_model_path  = vad_path;
    topts.threads         = threads;
    topts.word_timestamps = word_ts;
    topts.verbose         = verbose;
    std::vector<transcribe::VadRegion> vad_regions;
    if (debug) topts.vad_regions = &vad_regions;

    logging::info("transcribing");
    std::vector<srt::Segment> segments;
    if (!transcribe::run(pcm, topts, segments, err)) {
        logging::error("transcribe failed: " + err);
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    logging::logf("INFO", "transcribed %zu segments", segments.size());

    // Optional constant sync shift. Whisper's segment timestamps can lead or lag
    // the true audio (typically ~0.4s in the VAD path); this lets the user nudge
    // every cue by a fixed amount. Positive = later, negative = earlier.
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
        di.model = model; di.isolate = isolate; di.center = center; di.vad = vad; di.word_ts = word_ts;
        std::string dbg_path = output + ".debug.txt";
        std::string derr;
        if (debugreport::write(dbg_path, pcm, segments, vad_regions, refp, di, derr)) {
            std::fprintf(stderr, "[srt] wrote debug report: %s\n", dbg_path.c_str());
            logging::logf("INFO", "wrote debug report to %s", dbg_path.c_str());
        } else {
            std::fprintf(stderr, "warning: debug report failed (%s)\n", derr.c_str());
        }
    }
    return 0;
}
