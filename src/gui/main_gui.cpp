// SRTCreator GUI - minimal window with drag-and-drop, explicit input/output
// paths (with Browse), option toggles, a live transcript panel and progress bar.

#ifndef UNICODE
#define UNICODE
#endif
#include <windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shellapi.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "audio.h"
#include "transcribe.h"
#include "srt.h"
#include "models.h"
#include "separate.h"
#include "log.h"
#include "debug.h"
#include "timeline.h"
#include "segment_mode.h"

static std::wstring format_duration_hms_w(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    long long total_s = (long long)(seconds + 0.5);
    long long h = total_s / 3600;
    long long m = (total_s % 3600) / 60;
    long long s = total_s % 60;
    wchar_t buf[32];
    swprintf_s(buf, L"%02lld:%02lld:%02lld", h, m, s);
    return std::wstring(buf);
}

static std::string format_duration_hms(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    long long total_s = (long long)(seconds + 0.5);
    long long h = total_s / 3600;
    long long m = (total_s % 3600) / 60;
    long long s = total_s % 60;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld", h, m, s);
    return std::string(buf);
}

#pragma comment(linker, "\"/manifestdependency:type='win32' \
name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "shell32.lib")

#define WM_APP_SEGMENT  (WM_APP + 1) // lParam = wchar_t* : append line
#define WM_APP_PROGRESS (WM_APP + 2) // wParam = 0..100
#define WM_APP_STATUS   (WM_APP + 3) // lParam = wchar_t*
#define WM_APP_MARQUEE  (WM_APP + 4) // wParam = 1 start / 0 stop
#define WM_APP_DONE     (WM_APP + 5) // lParam = wchar_t* ; wParam = success

enum {
    ID_TRANSLATE = 1001, ID_CENTER, ID_DUMP, ID_DEBUG, ID_MULTIPASS,
    ID_MODEL, ID_LANG, ID_VMODEL, ID_EDIT, ID_PROGRESS, ID_STATUS,
    ID_INPUT, ID_INPUT_BROWSE, ID_OUTPUT, ID_OUTPUT_BROWSE, ID_START
};

static HWND g_status;
static HWND g_input_lbl, g_input, g_input_browse, g_output_lbl, g_output, g_output_browse;
static HWND g_translate, g_center, g_dump, g_debug, g_multipass;
static HWND g_model_lbl, g_model, g_lang_lbl, g_lang, g_vmodel_lbl, g_vmodel, g_start;
static HWND g_edit, g_progress;
static HWND g_tip = nullptr; // shared tooltip control for the checkboxes
static std::atomic<bool> g_running{false};

static std::wstring to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}
static std::string to_utf8(const std::wstring& w) {
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
static void post_str(HWND h, UINT msg, WPARAM wp, const std::wstring& s) {
    PostMessageW(h, msg, wp, (LPARAM)_wcsdup(s.c_str()));
}
static bool checked(HWND h) { return SendMessageW(h, BM_GETCHECK, 0, 0) == BST_CHECKED; }
static std::wstring get_text(HWND h) {
    int n = GetWindowTextLengthW(h);
    std::wstring s(n, L'\0');
    GetWindowTextW(h, s.data(), n + 1);
    return s;
}
static void append_line(HWND hEdit, const wchar_t* text) {
    int len = GetWindowTextLengthW(hEdit);
    SendMessageW(hEdit, EM_SETSEL, len, len);
    SendMessageW(hEdit, EM_REPLACESEL, FALSE, (LPARAM)text);
}
static std::wstring derive_srt(std::wstring p) {
    size_t slash = p.find_last_of(L"/\\");
    size_t dot   = p.find_last_of(L'.');
    return (dot == std::wstring::npos || (slash != std::wstring::npos && dot < slash))
               ? p + L".srt" : p.substr(0, dot) + L".srt";
}
static std::wstring basename_of(const std::wstring& p) {
    size_t s = p.find_last_of(L"/\\");
    return s == std::wstring::npos ? p : p.substr(s + 1);
}

// // ---- job ----
struct Job {
    HWND hwnd;
    std::wstring input, output;
    std::string model, language, vocal_model;
    bool translate, flash, word_ts, center, infill, dump_audio, debug, segmode;
    int max_line_length;
};

static void do_job(Job job) {
    HWND hwnd = job.hwnd;
    std::string input = to_utf8(job.input);
    std::string err;

    logging::logf("INFO", "job: input=%s out=%s model=%s lang=%s translate=%d center=%d infill=%d wrap=%d",
                  input.c_str(), to_utf8(job.output).c_str(), job.model.c_str(), job.language.c_str(),
                  (int)job.translate, (int)job.center, (int)job.infill, job.max_line_length);

    std::string models_dir = models::default_models_dir();
    auto dl = [hwnd](int pct) { PostMessageW(hwnd, WM_APP_PROGRESS, (WPARAM)pct, 0); };

    post_str(hwnd, WM_APP_STATUS, 0, L"Preparing Whisper model…");
    PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
    std::string model_path;
    if (!models::resolve(job.model, models_dir, true, model_path, err, dl)) {
        logging::error("model resolve failed: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }

    std::string vad_path;
    post_str(hwnd, WM_APP_STATUS, 0, L"Preparing VAD model…");
    if (!models::ensure_vad(models_dir, true, vad_path, err, dl)) {
        logging::error("VAD model unavailable: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }

    auto t_job_start = std::chrono::steady_clock::now();
    double dur_phase1 = 0.0; // Audio/center channel extraction
    double dur_phase2 = 0.0; // Voice isolation
    double dur_phase3 = 0.0; // Speech transcription (Pass 1)
    double dur_phase4 = 0.0; // Silence sanitization (Pass 2)
    double dur_phase5 = 0.0; // Targeted audio infill (Pass 3)

    PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
    auto prog = [hwnd](int pct) { PostMessageW(hwnd, WM_APP_PROGRESS, (WPARAM)pct, 0); };
    std::vector<float> pcm;
    std::vector<float> mix;
    std::vector<float> vocals;
    std::vector<srt::Segment> segments;
    std::vector<transcribe::VadRegion> vad_regions;
    timeline::TimelineMap tl_map;

    struct JobCleanup {
        std::vector<float>& pcm;
        std::vector<float>& mix;
        std::vector<float>& vocals;
        std::vector<srt::Segment>& segments;
        std::vector<transcribe::VadRegion>& vad_regions;
        timeline::TimelineMap& tl_map;
        ~JobCleanup() {
            std::vector<float>().swap(pcm);
            std::vector<float>().swap(mix);
            std::vector<float>().swap(vocals);
            std::vector<srt::Segment>().swap(segments);
            std::vector<transcribe::VadRegion>().swap(vad_regions);
            tl_map.intervals.clear(); tl_map.intervals.shrink_to_fit();
            tl_map.actions.clear(); tl_map.actions.shrink_to_fit();
            tl_map.missing_vocal.clear(); tl_map.missing_vocal.shrink_to_fit();
            SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
            logging::info("memory: cleaned up all audio buffers and trimmed working set");
        }
    } cleanup{pcm, mix, vocals, segments, vad_regions, tl_map};

    // Cache: reuse a 16k mono whisper input sitting next to the movie (a prior run's
    // dump, or the CLI's) to skip decode + isolation entirely - mirrors main.cpp
    // (see §14). Prefer the isolated vocal stem. This is what the GUI was missing.
    std::string cached_wav;
    {
        size_t cdot = input.find_last_of('.');
        std::string cbase = (cdot == std::string::npos) ? input : input.substr(0, cdot);
        for (const char* suf : { ".vocals16k.wav", ".whisper16k.wav" }) {
            std::string cand = cbase + suf;
            FILE* cf = std::fopen(cand.c_str(), "rb");
            if (cf) { std::fclose(cf); cached_wav = cand; break; }
        }
    }

    if (!cached_wav.empty()) {
        post_str(hwnd, WM_APP_STATUS, 0, L"Using cached audio (skipping decode + isolation)…");
        post_str(hwnd, WM_APP_SEGMENT, 0, L"[cache] using " + to_wide(cached_wav) + L"\r\n");
        logging::logf("INFO", "using cached whisper input %s (skip decode/isolation)", cached_wav.c_str());
        auto t_c0 = std::chrono::steady_clock::now();
        audio::DecodeOptions dc;
        dc.sample_rate = 16000; dc.channels = 1; dc.center_channel_only = false;
        dc.on_progress = prog;
        if (!audio::decode(cached_wav, dc, pcm, err)) {
            logging::error("cached audio load failed: " + err);
            post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
        }
        dur_phase1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_c0).count();
        logging::logf("INFO", "Phase 1 (Cached audio load) completed in %s (%.2fs)",
                      format_duration_hms(dur_phase1).c_str(), dur_phase1);
        post_str(hwnd, WM_APP_SEGMENT, 0,
                 L"[timing] Phase 1 (Cached audio load): " + format_duration_hms_w(dur_phase1) + L"\r\n");
        post_str(hwnd, WM_APP_SEGMENT, 0,
                 L"[cache] audio ready: " + std::to_wstring((int)(pcm.size() / 16000.0 / 60.0)) + L" min\r\n");
    } else {
    post_str(hwnd, WM_APP_STATUS, 0, L"Decoding (44.1k stereo for separation)…");
    audio::DecodeOptions d;
    d.sample_rate = 44100; d.channels = 2; d.center_channel_only = job.center;
    d.on_progress = prog;
    auto t_p1_start = std::chrono::steady_clock::now();
    if (!audio::decode(input, d, mix, err)) {
        logging::error("decode failed: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
    }
    auto t_p1_end = std::chrono::steady_clock::now();
    dur_phase1 = std::chrono::duration<double>(t_p1_end - t_p1_start).count();
    logging::logf("INFO", "Phase 1 (Audio extraction) completed in %s (%.2fs)",
                  format_duration_hms(dur_phase1).c_str(), dur_phase1);
    post_str(hwnd, WM_APP_SEGMENT, 0,
             L"[timing] Phase 1 (Audio extraction): " + format_duration_hms_w(dur_phase1) + L"\r\n");

    std::string vpath; separate::Params sp;
    post_str(hwnd, WM_APP_STATUS, 0, L"Preparing vocal model…");
    if (!separate::ensure_model(job.vocal_model, models_dir, true, vpath, sp, err, prog)) {
        logging::error("vocal model: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
    }
    PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
    post_str(hwnd, WM_APP_STATUS, 0, L"Isolating vocals (removing music)…");
    logging::logf("INFO", "isolating vocals with %s", job.vocal_model.c_str());
    auto t_p2_start = std::chrono::steady_clock::now();
    if (!separate::isolate_vocals(mix, vpath, sp, vocals, prog, err)) {
        logging::error("separation failed: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
    }
    std::vector<float>().swap(mix); // free the 44.1k stereo mix before resampling
    if (!audio::resample_to_mono(vocals, 44100, 1, 16000, pcm, err)) {
        logging::error("resample failed: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
    }
    std::vector<float>().swap(vocals); // IMMEDIATELY free 44.1k mono vocals (~1.6GB)
    auto t_p2_end = std::chrono::steady_clock::now();
    dur_phase2 = std::chrono::duration<double>(t_p2_end - t_p2_start).count();
    logging::logf("INFO", "Phase 2 (Voice isolation) completed in %s (%.2fs)",
                  format_duration_hms(dur_phase2).c_str(), dur_phase2);
    post_str(hwnd, WM_APP_SEGMENT, 0,
             L"[timing] Phase 2 (Voice isolation): " + format_duration_hms_w(dur_phase2) + L"\r\n");
    logging::logf("INFO", "isolated audio ready %.1f min", pcm.size() / 16000.0 / 60.0);
    } // end else (no cache): decode + isolation

    // Dump the exact 16 kHz mono track fed to whisper, next to the input - ONLY
    // when the Dump audio box is checked.
    if (job.dump_audio) {
        std::string dump = input;
        size_t dot = dump.find_last_of('.');
        dump = (dot == std::string::npos ? dump : dump.substr(0, dot)) + ".vocals16k.wav";
        std::string werr;
        if (audio::write_wav(dump, pcm, 16000, 1, werr)) {
            logging::logf("INFO", "dumped whisper input audio to %s", dump.c_str());
            post_str(hwnd, WM_APP_SEGMENT, 0, L"[debug] wrote audio: " + to_wide(dump) + L"\r\n");
        } else {
            logging::error("audio dump failed: " + werr);
        }
    }

    PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
    post_str(hwnd, WM_APP_STATUS, 0, L"Transcribing…");
    logging::info("transcribing");
    transcribe::Options t;
    t.model_path      = model_path;
    t.language        = job.language;
    t.translate       = job.translate;
    t.flash_attn      = job.flash;
    t.vad             = true;
    t.vad_model_path  = vad_path;
    t.word_timestamps = job.word_ts;
    t.vad_regions     = &vad_regions;
    t.on_progress = [hwnd](int pct) { PostMessageW(hwnd, WM_APP_PROGRESS, (WPARAM)pct, 0); };
    // Dedup the live transcript the same way the SRT writer does, so the scrolling
    // panel doesn't fill with whisper's repeated hallucination lines.
    srt::Deduper live_dedup;
    t.on_segment  = [hwnd, &live_dedup](const srt::Segment& s) {
        if (live_dedup.is_duplicate(s.text)) return;
        std::wstring line = L"[" + to_wide(srt::format_timestamp(s.t0)).substr(0, 8) + L"]  "
                          + to_wide(s.text) + L"\r\n";
        post_str(hwnd, WM_APP_SEGMENT, 0, line);
    };

    transcribe::Session session;
    auto t_p3_start = std::chrono::steady_clock::now();
    if (!session.init(t, err)) {
        logging::error("whisper init failed: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }
    if (job.segmode) {
        // EXPERIMENTAL: VAD-segmented per-region transcription (see segment_mode.*).
        // Leaves vad_regions empty so the Pass 2/3 block below is skipped. Progress
        // and live cues are driven by segment_mode's region-level callbacks (the same
        // progress bar / transcript panel the default path uses).
        post_str(hwnd, WM_APP_STATUS, 0, L"Transcribing (segment mode)…");
        post_str(hwnd, WM_APP_SEGMENT, 0, L"[segment] VAD-segmented per-region transcription\r\n");
        segment_mode::Options sopts;
        sopts.vad_model_path = vad_path;
        if (!segment_mode::run(pcm, session, t, sopts, segments, err,
                [hwnd](const std::string& msg){ post_str(hwnd, WM_APP_SEGMENT, 0, L"[segment] " + to_wide(msg) + L"\r\n"); },
                [hwnd](int pct){ PostMessageW(hwnd, WM_APP_PROGRESS, (WPARAM)pct, 0); },
                [hwnd, &live_dedup](const srt::Segment& s){
                    if (live_dedup.is_duplicate(s.text)) return;
                    std::wstring line = L"[" + to_wide(srt::format_timestamp(s.t0)).substr(0, 8) + L"]  "
                                      + to_wide(s.text) + L"\r\n";
                    post_str(hwnd, WM_APP_SEGMENT, 0, line);
                })) {
            logging::error("segment-mode failed: " + err);
            session.close();
            post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
            return;
        }
    } else if (!session.transcribe(pcm, t, segments, err)) {
        logging::error("transcribe failed: " + err);
        session.close();
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }

    auto t_p3_end = std::chrono::steady_clock::now();
    dur_phase3 = std::chrono::duration<double>(t_p3_end - t_p3_start).count();
    logging::logf("INFO", "Phase 3 (Transcription) completed in %s (%.2fs)",
                  format_duration_hms(dur_phase3).c_str(), dur_phase3);
    post_str(hwnd, WM_APP_SEGMENT, 0,
             L"[timing] Phase 3 (Transcription): " + format_duration_hms_w(dur_phase3) + L"\r\n");
    logging::logf("INFO", "transcribed %zu raw segments", segments.size());

    // Pass 2: Sanitize cues & split mid-sentence pauses against timeline
    size_t orig_cue_count = segments.size();
    std::wstring out = job.output.empty() ? derive_srt(job.input) : job.output;
    std::string out_utf8 = to_utf8(out);
    size_t dot_pos = out_utf8.find_last_of('.');
    std::string json_path = (dot_pos == std::string::npos ? out_utf8 : out_utf8.substr(0, dot_pos)) + ".timeline.json";

    if (!job.segmode && !vad_regions.empty()) {
        post_str(hwnd, WM_APP_STATUS, 0, L"Pass 2: Sanitizing cues against timeline…");
        auto t_p4_start = std::chrono::steady_clock::now();
        tl_map = timeline::build_timeline(input, pcm, vad_regions);
        timeline::eliminate_boundary_bleeds(segments, tl_map, pcm, session, t, [hwnd](const std::string& msg) {
            post_str(hwnd, WM_APP_SEGMENT, 0, L"[bleed] " + to_wide(msg) + L"\r\n");
        });
        timeline::sanitize_and_split(segments, tl_map, 1.5);
        timeline::find_missing_vocal_regions(segments, tl_map, 0.8, 0.20);
        auto t_p4_end = std::chrono::steady_clock::now();
        dur_phase4 = std::chrono::duration<double>(t_p4_end - t_p4_start).count();
        logging::logf("INFO", "Phase 4 (Silence sanitization) completed in %s (%.2fs)",
                      format_duration_hms(dur_phase4).c_str(), dur_phase4);
        post_str(hwnd, WM_APP_SEGMENT, 0,
                 L"[timing] Phase 4 (Silence sanitization): " + format_duration_hms_w(dur_phase4) + L"\r\n");

        auto ac_p2 = timeline::count_actions(tl_map);
        logging::logf("INFO", "Pass 2 sanitation: %d actions applied (cues: %zu -> %zu)",
                      ac_p2.total(), orig_cue_count, segments.size());
        post_str(hwnd, WM_APP_SEGMENT, 0,
                 L"[timeline] Pass 2 applied " + std::to_wstring(ac_p2.total()) +
                 L" timing fixes (cues: " + std::to_wstring(orig_cue_count) + L" -> " +
                 std::to_wstring(segments.size()) + L")\r\n");

        if (job.infill && !tl_map.missing_vocal.empty()) {
            post_str(hwnd, WM_APP_STATUS, 0, L"Pass 3: Infilling missed speech…");
            post_str(hwnd, WM_APP_SEGMENT, 0,
                     L"[timeline] Pass 3 checking " + std::to_wstring(tl_map.missing_vocal.size()) +
                     L" potential missed speech regions...\r\n");

            auto t_p5_start = std::chrono::steady_clock::now();
            timeline::infill_missing_regions(pcm, session, t, tl_map, segments, [hwnd](const std::string& msg) {
                post_str(hwnd, WM_APP_SEGMENT, 0, L"[infill] " + to_wide(msg) + L"\r\n");
            });
            auto t_p5_end = std::chrono::steady_clock::now();
            dur_phase5 = std::chrono::duration<double>(t_p5_end - t_p5_start).count();
            logging::logf("INFO", "Phase 5 (Targeted audio infill) completed in %s (%.2fs)",
                          format_duration_hms(dur_phase5).c_str(), dur_phase5);
            post_str(hwnd, WM_APP_SEGMENT, 0,
                     L"[timing] Phase 5 (Targeted audio infill): " + format_duration_hms_w(dur_phase5) + L"\r\n");
        }

        std::string jerr;
        if (timeline::write_json(tl_map, json_path, jerr)) {
            logging::info("wrote timeline json: " + json_path);
            post_str(hwnd, WM_APP_SEGMENT, 0, L"[timeline] wrote analysis: " + to_wide(json_path) + L"\r\n");
        } else {
            logging::error("failed to write timeline json: " + jerr);
        }
    }

    session.close(); // explicitly release Whisper model and reset CUDA device memory

    auto t_job_end = std::chrono::steady_clock::now();
    double dur_total = std::chrono::duration<double>(t_job_end - t_job_start).count();
    auto ac = timeline::count_actions(tl_map);

    // Log complete timing summary
    logging::logf("INFO", "Timing Summary: 1.Audio extraction=%s (%.2fs) | 2.Voice isolation=%s (%.2fs) | 3.Transcription=%s (%.2fs) | 4.Silence sanitize=%s (%.2fs) | 5.Targeted infill=%s | Total=%s (%.2fs)",
                  format_duration_hms(dur_phase1).c_str(), dur_phase1,
                  format_duration_hms(dur_phase2).c_str(), dur_phase2,
                  format_duration_hms(dur_phase3).c_str(), dur_phase3,
                  format_duration_hms(dur_phase4).c_str(), dur_phase4,
                  job.infill ? (tl_map.missing_vocal.empty() ? "N/A (0 missed)" : (format_duration_hms(dur_phase5) + " (" + std::to_string((int)dur_phase5) + "s)").c_str()) : "N/A (disabled)",
                  format_duration_hms(dur_total).c_str(), dur_total);

    logging::logf("INFO", "Post-Processing Summary: Vocal coverage=%.1f%% (%.1fs vocal / %.1fs silence) | Pass 2 fixes=%d (clamped=%d, snapped=%d, split=%d, dropped=%d, bleeds=%d) | Pass 2.5 missed gaps=%zu | Pass 3 infilled=%d | Pass 3 re-anchored=%d",
                  tl_map.analysis.vocal_coverage_pct, tl_map.analysis.total_vocal_sec, tl_map.analysis.total_silence_sec,
                  ac.total() - ac.infilled - ac.reanchored, ac.clamped, ac.snapped, ac.split, ac.dropped, ac.bleeds_pruned,
                  tl_map.missing_vocal.size(), ac.infilled, ac.reanchored);

    // Append Timing Summary and Post-Processing Summary block to scrolling subtitle area (g_edit)
    std::wstring summary =
        L"\r\n----------------------------------------\r\n"
        L"Timing Summary:\r\n"
        L"  1. Audio extraction:      " + format_duration_hms_w(dur_phase1) + L"\r\n"
        L"  2. Voice isolation:       " + format_duration_hms_w(dur_phase2) + L"\r\n"
        L"  3. Speech transcription:  " + format_duration_hms_w(dur_phase3) + L"\r\n"
        L"  4. Silence sanitization:  " + format_duration_hms_w(dur_phase4) + L"\r\n"
        L"  5. Targeted audio infill: " + (job.infill ? (tl_map.missing_vocal.empty() ? L"00:00:00 (0 missed)" : format_duration_hms_w(dur_phase5)) : L"N/A (disabled)") + L"\r\n"
        L"  Total time taken:         " + format_duration_hms_w(dur_total) + L"\r\n"
        L"----------------------------------------\r\n"
        L"Post-Processing Summary (Pass 2 & 3):\r\n"
        L"  Vocal coverage:    " + std::to_wstring((int)(tl_map.analysis.vocal_coverage_pct + 0.5)) + L"% (" +
        std::to_wstring((int)(tl_map.analysis.total_vocal_sec + 0.5)) + L"s vocal / " +
        std::to_wstring((int)(tl_map.analysis.total_silence_sec + 0.5)) + L"s silence)\r\n"
        L"  Cues processed:    " + std::to_wstring(orig_cue_count) + L" -> " + std::to_wstring(segments.size()) + L"\r\n"
        L"  Pass 2 fixes:      " + std::to_wstring(ac.total() - ac.infilled - ac.reanchored) + L" applied\r\n"
        L"    - Pruned bleeds:      " + std::to_wstring(ac.bleeds_pruned) + L"\r\n"
        L"    - Clamped trailing:   " + std::to_wstring(ac.clamped) + L"\r\n"
        L"    - Snapped leading:    " + std::to_wstring(ac.snapped) + L"\r\n"
        L"    - Split pauses:       " + std::to_wstring(ac.split) + L"\r\n"
        L"    - Dropped halluc.:    " + std::to_wstring(ac.dropped) + L"\r\n"
        L"  Missed vocal gaps: " + std::to_wstring(tl_map.missing_vocal.size()) + L" detected\r\n"
        L"  Pass 3 infilled:   " + std::to_wstring(ac.infilled) + L" recovered cues\r\n"
        L"  Pass 3 re-anchored:" + std::to_wstring(ac.reanchored) + L" displaced cues\r\n"
        L"----------------------------------------\r\n\r\n";
    post_str(hwnd, WM_APP_SEGMENT, 0, summary);

    if (srt::write(segments, to_utf8(out), job.max_line_length, err)) {
        PostMessageW(hwnd, WM_APP_PROGRESS, 100, 0);
        logging::info("wrote " + to_utf8(out));
        // Timing diagnostics next to the SRT (same folder as the output).
        if (job.debug) {
            debugreport::Info di;
            di.model = job.model; di.isolate = true; di.center = job.center;
            di.vad = true; di.word_ts = job.word_ts;
            std::string dbg = to_utf8(out) + ".debug.txt";
            std::string derr;
            if (debugreport::write(dbg, pcm, segments, vad_regions, nullptr, di, derr)) {
                logging::info("wrote debug report " + dbg);
                post_str(hwnd, WM_APP_SEGMENT, 0, L"[debug] wrote report: " + to_wide(dbg) + L"\r\n");
            } else {
                logging::error("debug report failed: " + derr);
            }
        }
        post_str(hwnd, WM_APP_DONE, 1, L"Saved: " + out + L" (Total: " + format_duration_hms_w(dur_total) + L")");
        return;
    }
    logging::error("write failed at " + to_utf8(out) + ": " + err);
    const char* up = std::getenv("USERPROFILE");
    if (up && *up) {
        std::wstring fb = to_wide(std::string(up)) + L"\\Desktop\\" + basename_of(out);
        std::string ferr;
        if (srt::write(segments, to_utf8(fb), job.max_line_length, ferr)) {
            PostMessageW(hwnd, WM_APP_PROGRESS, 100, 0);
            logging::info("wrote fallback " + to_utf8(fb));
            post_str(hwnd, WM_APP_DONE, 1, L"Output folder not writable. Saved to Desktop: " + fb + L" (Total: " + format_duration_hms_w(dur_total) + L")");
            return;
        }
    }
    post_str(hwnd, WM_APP_DONE, 0, L"Error writing SRT: " + to_wide(err));
}

static void set_busy(bool busy); // greys out controls during a run

static void run_job(Job job) {
    HWND hwnd = job.hwnd;
    try { do_job(job); }
    catch (const std::exception& e) {
        logging::error(std::string("exception: ") + e.what());
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(e.what()));
    } catch (...) {
        logging::error("unknown exception in worker");
        post_str(hwnd, WM_APP_DONE, 0, L"Error: unknown failure (see log)");
    }
}

static void start_job(HWND hwnd) {
    std::wstring input = get_text(g_input);
    if (input.empty()) { SetWindowTextW(g_status, L"Choose an input file first (drag a file or Browse)."); return; }
    if (GetFileAttributesW(input.c_str()) == INVALID_FILE_ATTRIBUTES) {
        SetWindowTextW(g_status, L"Input file not found."); return;
    }
    if (g_running.exchange(true)) return;

    std::wstring output = get_text(g_output);
    if (output.empty()) { output = derive_srt(input); SetWindowTextW(g_output, output.c_str()); }

    SetWindowTextW(g_edit, L"");
    logging::info("start: " + to_utf8(input));

    Job job;
    job.hwnd = hwnd; job.input = input; job.output = output;
    job.model       = to_utf8(get_text(g_model));
    job.language    = to_utf8(get_text(g_lang));
    job.vocal_model = to_utf8(get_text(g_vmodel));
    job.translate   = checked(g_translate);
    job.flash       = true;   // always on
    job.word_ts     = true;   // always on (DTW word timing)
    job.center      = checked(g_center);
    job.infill      = true;   // always on; effective in multi-pass mode only
    job.dump_audio  = checked(g_dump);
    job.debug       = checked(g_debug);
    // Segment mode is the default; the checkbox opts INTO the legacy multi-pass pipeline.
    job.segmode     = !checked(g_multipass);
    job.max_line_length = 42; // line wrapping always on
    set_busy(true);
    std::thread(run_job, job).detach();
}

static void set_input_path(const std::wstring& path) {
    SetWindowTextW(g_input, path.c_str());
    SetWindowTextW(g_output, derive_srt(path).c_str()); // keep output in sync
}

// ---- file dialogs ----
static void browse_input(HWND hwnd) {
    wchar_t buf[MAX_PATH] = {0};
    OPENFILENAMEW ofn{}; ofn.lStructSize = sizeof(ofn); ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = L"Media files\0*.mp4;*.mkv;*.avi;*.mov;*.ts;*.m4v;*.webm;*.flv;*.wav;*.mp3;*.flac;*.m4a;*.aac;*.ac3;*.eac3;*.dts\0All files\0*.*\0";
    ofn.lpstrFile = buf; ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (GetOpenFileNameW(&ofn)) set_input_path(buf);
}
static void browse_output(HWND hwnd) {
    wchar_t buf[MAX_PATH] = {0};
    std::wstring cur = get_text(g_output);
    if (!cur.empty()) wcsncpy_s(buf, cur.c_str(), _TRUNCATE);
    OPENFILENAMEW ofn{}; ofn.lStructSize = sizeof(ofn); ofn.hwndOwner = hwnd;
    ofn.lpstrFilter = L"SubRip subtitle\0*.srt\0All files\0*.*\0";
    ofn.lpstrFile = buf; ofn.nMaxFile = MAX_PATH;
    ofn.lpstrDefExt = L"srt";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
    if (GetSaveFileNameW(&ofn)) SetWindowTextW(g_output, buf);
}

// ---- layout ----
static void layout(HWND hwnd) {
    RECT rc; GetClientRect(hwnd, &rc);
    int W = rc.right, H = rc.bottom, m = 12, bw = 80, lw = 55;
    int y = m;

    MoveWindow(g_input_lbl, m, y + 3, lw, 18, TRUE);
    MoveWindow(g_input, m + lw, y, W - 2 * m - lw - bw - 6, 24, TRUE);
    MoveWindow(g_input_browse, W - m - bw, y, bw, 24, TRUE); y += 30;

    MoveWindow(g_output_lbl, m, y + 3, lw, 18, TRUE);
    MoveWindow(g_output, m + lw, y, W - 2 * m - lw - bw - 6, 24, TRUE);
    MoveWindow(g_output_browse, W - m - bw, y, bw, 24, TRUE); y += 32;

    // Checkbox row 1: pipeline options.
    int x = m;
    struct { HWND h; int w; } r1[] = {
        {g_translate, 170}, {g_center, 210}, {g_multipass, 210}
    };
    for (auto& c : r1) { MoveWindow(c.h, x, y, c.w, 24, TRUE); x += c.w + 16; }
    y += 28;
    // Checkbox row 2: debug options.
    x = m;
    struct { HWND h; int w; } r2[] = {
        {g_dump, 175}, {g_debug, 150}
    };
    for (auto& c : r2) { MoveWindow(c.h, x, y, c.w, 24, TRUE); x += c.w + 16; }
    y += 32;

    // Model / language / vocal-model / Start.
    MoveWindow(g_model_lbl, m, y + 4, 45, 18, TRUE);
    MoveWindow(g_model, m + 48, y, 190, 300, TRUE);
    int lx = m + 48 + 190 + 16;
    MoveWindow(g_lang_lbl, lx, y + 4, 38, 18, TRUE);
    MoveWindow(g_lang, lx + 40, y, 90, 300, TRUE);
    int vx = lx + 40 + 90 + 16;
    MoveWindow(g_vmodel_lbl, vx, y + 4, 44, 18, TRUE);
    MoveWindow(g_vmodel, vx + 46, y, 190, 300, TRUE);
    MoveWindow(g_start, W - m - 110, y, 110, 26, TRUE);
    y += 34;

    // Transcript panel, then status just above the progress bar at the bottom.
    int prog_h = 22, status_h = 18;
    int bottom = H - m - prog_h - 4 - status_h;
    MoveWindow(g_edit, m, y, W - 2 * m, bottom - y, TRUE);
    MoveWindow(g_status, m, H - m - prog_h - 4 - status_h, W - 2 * m, status_h, TRUE);
    MoveWindow(g_progress, m, H - m - prog_h, W - 2 * m, prog_h, TRUE);
}

// Register a hover tooltip for a control. TTF_SUBCLASS lets the tooltip control
// handle the hover/hide itself, so it appears on hover and auto-hides after a delay.
static void add_tip(HWND parent, HWND ctrl, const wchar_t* text) {
    if (!g_tip || !ctrl) return;
    TOOLINFOW ti = {};
    ti.cbSize   = sizeof(ti);
    ti.uFlags   = TTF_IDISHWND | TTF_SUBCLASS;
    ti.hwnd     = parent;
    ti.uId      = (UINT_PTR)ctrl;
    ti.lpszText = const_cast<wchar_t*>(text);
    SendMessageW(g_tip, TTM_ADDTOOLW, 0, (LPARAM)&ti);
}

static void create_tooltips(HWND hwnd) {
    g_tip = CreateWindowExW(WS_EX_TOPMOST, TOOLTIPS_CLASSW, nullptr,
                            WS_POPUP | TTS_ALWAYSTIP | TTS_NOPREFIX,
                            CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
                            hwnd, nullptr,
                            (HINSTANCE)GetWindowLongPtrW(hwnd, GWLP_HINSTANCE), nullptr);
    if (!g_tip) return;
    SendMessageW(g_tip, TTM_SETMAXTIPWIDTH, 0, 380);            // enable multi-line wrapping
    SendMessageW(g_tip, TTM_SETDELAYTIME, TTDT_AUTOPOP, (LPARAM)15000); // keep visible ~15s to read
    add_tip(hwnd, g_translate, L"Translate speech to English instead of transcribing in the original language.");
    add_tip(hwnd, g_center,    L"Extract only the front-center channel (where dialogue sits in 5.1/7.1 mixes) before vocal isolation. Turn off to isolate from the full stereo downmix.");
    add_tip(hwnd, g_dump,      L"Write the exact 16 kHz mono audio fed to Whisper next to the input (<stem>.vocals16k.wav), for debugging.");
    add_tip(hwnd, g_debug,     L"Write a detailed timing/diagnostics report next to the output SRT.");
    add_tip(hwnd, g_multipass, L"Use the legacy multi-pass timeline pipeline: transcribe the whole track at once, then sanitize timing and infill gaps. Leave unchecked for the default segment mode, which transcribes each detected speech region in isolation for tighter timing.");
}

static HWND mk(HWND parent, const wchar_t* cls, const wchar_t* text, DWORD style, int id) {
    return CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style, 0, 0, 0, 0,
                           parent, (HMENU)(INT_PTR)id,
                           (HINSTANCE)GetWindowLongPtrW(parent, GWLP_HINSTANCE), nullptr);
}

static void create_controls(HWND hwnd) {
    g_status = mk(hwnd, L"STATIC", L"Drag a movie/audio file here, or Browse - then Start.", SS_LEFT, ID_STATUS);

    g_input_lbl    = mk(hwnd, L"STATIC", L"Input:", SS_LEFT, 0);
    g_input        = mk(hwnd, L"EDIT", L"", ES_AUTOHSCROLL | WS_BORDER, ID_INPUT);
    g_input_browse = mk(hwnd, L"BUTTON", L"Browse…", BS_PUSHBUTTON, ID_INPUT_BROWSE);
    g_output_lbl   = mk(hwnd, L"STATIC", L"Output:", SS_LEFT, 0);
    g_output       = mk(hwnd, L"EDIT", L"", ES_AUTOHSCROLL | WS_BORDER, ID_OUTPUT);
    g_output_browse= mk(hwnd, L"BUTTON", L"Browse…", BS_PUSHBUTTON, ID_OUTPUT_BROWSE);

    // Only user-meaningful toggles are exposed. Flash attention, DTW word timestamps,
    // line wrapping (42) and Pass 3 infill are always on (infill applies in multi-pass
    // mode only) and are set unconditionally in the job, not shown as checkboxes.
    g_translate = mk(hwnd, L"BUTTON", L"Translate → English",       BS_AUTOCHECKBOX, ID_TRANSLATE);
    g_center    = mk(hwnd, L"BUTTON", L"Center channel (dialogue)", BS_AUTOCHECKBOX, ID_CENTER);
    g_dump      = mk(hwnd, L"BUTTON", L"Dump audio (debug)",        BS_AUTOCHECKBOX, ID_DUMP);
    g_debug     = mk(hwnd, L"BUTTON", L"Debug report",              BS_AUTOCHECKBOX, ID_DEBUG);
    g_multipass = mk(hwnd, L"BUTTON", L"Multi-pass mode (legacy)",  BS_AUTOCHECKBOX, ID_MULTIPASS);
    // Center channel OFF by default (user preference): isolate from the full stereo
    // downmix. Tick it to extract only the Front-Center channel first. The baseline
    // harness scores center-on slightly higher, but center-off captures more side
    // dialogue. Matches the CLI default (main.cpp `center=false`).

    g_model_lbl = mk(hwnd, L"STATIC", L"Model:", SS_LEFT, 0);
    g_model = mk(hwnd, L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL, ID_MODEL);
    for (auto s : { L"large-v3-turbo-q8_0", L"large-v3-turbo", L"large-v3",
                    L"medium", L"small", L"base", L"tiny",
                    L"small.en", L"base.en", L"tiny.en" })
        SendMessageW(g_model, CB_ADDSTRING, 0, (LPARAM)s);
    SendMessageW(g_model, CB_SETCURSEL, 0, 0);

    g_lang_lbl = mk(hwnd, L"STATIC", L"Lang:", SS_LEFT, 0);
    g_lang = mk(hwnd, L"COMBOBOX", L"", CBS_DROPDOWN | WS_VSCROLL, ID_LANG);
    for (auto s : { L"auto", L"en", L"es", L"fr", L"de", L"it", L"pt",
                    L"nl", L"ru", L"zh", L"ja", L"ko", L"hi", L"ar" })
        SendMessageW(g_lang, CB_ADDSTRING, 0, (LPARAM)s);
    SetWindowTextW(g_lang, L"auto");

    g_vmodel_lbl = mk(hwnd, L"STATIC", L"Vocal:", SS_LEFT, 0);
    g_vmodel = mk(hwnd, L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL, ID_VMODEL);
    for (const auto& mi : separate::registry())
        SendMessageW(g_vmodel, CB_ADDSTRING, 0, (LPARAM)to_wide(mi.name).c_str());
    SendMessageW(g_vmodel, CB_SETCURSEL, 0, 0);

    g_start = mk(hwnd, L"BUTTON", L"Start", BS_DEFPUSHBUTTON, ID_START);

    g_edit = mk(hwnd, L"EDIT", L"",
                ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_BORDER, ID_EDIT);
    SendMessageW(g_edit, EM_SETLIMITTEXT, 0, 0);
    g_progress = mk(hwnd, PROGRESS_CLASSW, L"", 0, ID_PROGRESS);
    SendMessageW(g_progress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));

    HFONT font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    for (HWND h : { g_status, g_input_lbl, g_input, g_input_browse, g_output_lbl, g_output,
                    g_output_browse, g_translate, g_center, g_dump, g_debug, g_multipass,
                    g_model_lbl, g_model, g_lang_lbl, g_lang, g_vmodel_lbl, g_vmodel,
                    g_start, g_edit })
        SendMessageW(h, WM_SETFONT, (WPARAM)font, TRUE);

    create_tooltips(hwnd); // hover help for every checkbox
}

static void set_busy(bool busy) {
    for (HWND h : { g_start, g_input_browse, g_output_browse, g_input, g_output,
                    g_translate, g_center, g_dump, g_debug, g_multipass,
                    g_model, g_lang, g_vmodel })
        EnableWindow(h, !busy);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE:  create_controls(hwnd); DragAcceptFiles(hwnd, TRUE); return 0;
    case WM_SIZE:    layout(hwnd); return 0;
    case WM_GETMINMAXINFO: {
        MINMAXINFO* mmi = (MINMAXINFO*)lp;
        mmi->ptMinTrackSize.x = 780; mmi->ptMinTrackSize.y = 480; return 0;
    }
    case WM_COMMAND: {
        switch (LOWORD(wp)) {
        case ID_INPUT_BROWSE:  if (!g_running) browse_input(hwnd);  return 0;
        case ID_OUTPUT_BROWSE: if (!g_running) browse_output(hwnd); return 0;
        case ID_START:         start_job(hwnd); return 0;
        }
        return 0;
    }
    case WM_DROPFILES: {
        if (!g_running) {
            wchar_t path[MAX_PATH];
            if (DragQueryFileW((HDROP)wp, 0, path, MAX_PATH)) {
                set_input_path(path);
                SetWindowTextW(g_status, L"Ready. Adjust options, then Start.");
            }
        }
        DragFinish((HDROP)wp);
        return 0;
    }
    case WM_APP_SEGMENT:  { wchar_t* s = (wchar_t*)lp; append_line(g_edit, s); free(s); return 0; }
    case WM_APP_PROGRESS: SendMessageW(g_progress, PBM_SETPOS, wp, 0); return 0;
    case WM_APP_STATUS:   { wchar_t* s = (wchar_t*)lp; SetWindowTextW(g_status, s); free(s); return 0; }
    case WM_APP_MARQUEE: {
        LONG_PTR st = GetWindowLongPtrW(g_progress, GWL_STYLE);
        if (wp) { SetWindowLongPtrW(g_progress, GWL_STYLE, st | PBS_MARQUEE); SendMessageW(g_progress, PBM_SETMARQUEE, TRUE, 30); }
        else    { SendMessageW(g_progress, PBM_SETMARQUEE, FALSE, 0); SetWindowLongPtrW(g_progress, GWL_STYLE, st & ~PBS_MARQUEE); }
        return 0;
    }
    case WM_APP_DONE: {
        wchar_t* s = (wchar_t*)lp; SetWindowTextW(g_status, s); free(s);
        g_running = false; set_busy(false);
        return 0;
    }
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

// Called from start_job path indirectly; disable controls when a run begins.
// (start_job sets g_running then spawns the worker; we grey out here.)

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int nShow) {
    SetProcessDPIAware();
    logging::init("gui");
    INITCOMMONCONTROLSEX icc{ sizeof(icc), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&icc);

    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc; wc.hInstance = hInst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = L"SRTCreatorWindow";
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(WS_EX_ACCEPTFILES, wc.lpszClassName, L"SRTCreator",
                                WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                                940, 700, nullptr, nullptr, hInst, nullptr);
    ShowWindow(hwnd, nShow);
    UpdateWindow(hwnd);

    // A file on the command line ("Open with…") pre-fills and starts.
    int argc = 0; LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (argv) {
        if (argc >= 2 && GetFileAttributesW(argv[1]) != INVALID_FILE_ATTRIBUTES) {
            set_input_path(argv[1]);
            start_job(hwnd);
        }
        LocalFree(argv);
    }

    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0)) { TranslateMessage(&m); DispatchMessageW(&m); }
    return 0;
}
