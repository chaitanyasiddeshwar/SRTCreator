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
    ID_TRANSLATE = 1001, ID_VAD, ID_FLASH, ID_WORDTS, ID_WRAP, ID_CENTER, ID_ISOLATE,
    ID_MODEL, ID_LANG, ID_VMODEL, ID_EDIT, ID_PROGRESS, ID_STATUS,
    ID_INPUT, ID_INPUT_BROWSE, ID_OUTPUT, ID_OUTPUT_BROWSE, ID_START
};

static HWND g_status;
static HWND g_input_lbl, g_input, g_input_browse, g_output_lbl, g_output, g_output_browse;
static HWND g_translate, g_vad, g_flash, g_wordts, g_wrap, g_center, g_isolate;
static HWND g_model_lbl, g_model, g_lang_lbl, g_lang, g_vmodel_lbl, g_vmodel, g_start;
static HWND g_edit, g_progress;
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

// ---- job ----
struct Job {
    HWND hwnd;
    std::wstring input, output;
    std::string model, language, vocal_model;
    bool translate, vad, flash, word_ts, center, isolate;
    int max_line_length;
};

static void do_job(Job job) {
    HWND hwnd = job.hwnd;
    std::string input = to_utf8(job.input);
    std::string err;

    logging::logf("INFO", "job: input=%s out=%s model=%s lang=%s translate=%d vad=%d center=%d wrap=%d",
                  input.c_str(), to_utf8(job.output).c_str(), job.model.c_str(), job.language.c_str(),
                  (int)job.translate, (int)job.vad, (int)job.center, job.max_line_length);

    std::string models_dir = models::default_models_dir();
    auto dl = [hwnd](int pct) { PostMessageW(hwnd, WM_APP_PROGRESS, (WPARAM)pct, 0); };

    post_str(hwnd, WM_APP_STATUS, 0, L"Preparing model (first run downloads it)…");
    PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
    std::string model_path;
    if (!models::resolve(job.model, models_dir, true, model_path, err, dl)) {
        logging::error("model resolve failed: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }

    std::string vad_path;
    bool vad = job.vad;
    if (vad) {
        post_str(hwnd, WM_APP_STATUS, 0, L"Preparing VAD model…");
        if (!models::ensure_vad(models_dir, true, vad_path, err, dl)) {
            logging::error("VAD unavailable, continuing without: " + err);
            vad = false;
        }
    }

    PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
    auto prog = [hwnd](int pct) { PostMessageW(hwnd, WM_APP_PROGRESS, (WPARAM)pct, 0); };
    std::vector<float> pcm;

    if (job.isolate) {
        post_str(hwnd, WM_APP_STATUS, 0, L"Decoding (44.1k stereo for separation)…");
        audio::DecodeOptions d;
        d.sample_rate = 44100; d.channels = 2; d.center_channel_only = job.center;
        d.on_progress = prog;
        std::vector<float> mix;
        if (!audio::decode(input, d, mix, err)) {
            logging::error("decode failed: " + err);
            post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
        }
        std::string vpath; separate::Params sp;
        post_str(hwnd, WM_APP_STATUS, 0, L"Preparing vocal model…");
        if (!separate::ensure_model(job.vocal_model, models_dir, true, vpath, sp, err, prog)) {
            logging::error("vocal model: " + err);
            post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
        }
        PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
        post_str(hwnd, WM_APP_STATUS, 0, L"Isolating vocals (removing music)…");
        logging::logf("INFO", "isolating vocals with %s", job.vocal_model.c_str());
        std::vector<float> vocals;
        if (!separate::isolate_vocals(mix, vpath, sp, vocals, prog, err)) {
            logging::error("separation failed: " + err);
            post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
        }
        std::vector<float>().swap(mix); // free the 44.1k stereo mix before resampling
        if (!audio::resample_to_mono(vocals, 44100, 1, 16000, pcm, err)) {
            logging::error("resample failed: " + err);
            post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
        }
    } else {
        post_str(hwnd, WM_APP_STATUS, 0, L"Decoding audio… (can take a while for long files)");
        logging::info("decoding audio");
        audio::DecodeOptions dopts;
        dopts.center_channel_only = job.center;
        dopts.on_progress = prog;
        if (!audio::decode(input, dopts, pcm, err)) {
            logging::error("decode failed: " + err);
            post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err)); return;
        }
    }
    logging::logf("INFO", "audio ready %.1f min", pcm.size() / 16000.0 / 60.0);

    PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
    post_str(hwnd, WM_APP_STATUS, 0, L"Transcribing…");
    logging::info("transcribing");
    std::vector<srt::Segment> segments;
    transcribe::Options t;
    t.model_path      = model_path;
    t.language        = job.language;
    t.translate       = job.translate;
    t.flash_attn      = job.flash;
    t.vad             = vad;
    t.vad_model_path  = vad_path;
    t.word_timestamps = job.word_ts;
    t.on_progress = [hwnd](int pct) { PostMessageW(hwnd, WM_APP_PROGRESS, (WPARAM)pct, 0); };
    t.on_segment  = [hwnd](const srt::Segment& s) {
        std::wstring line = L"[" + to_wide(srt::format_timestamp(s.t0)).substr(0, 8) + L"]  "
                          + to_wide(s.text) + L"\r\n";
        post_str(hwnd, WM_APP_SEGMENT, 0, line);
    };
    if (!transcribe::run(pcm, t, segments, err)) {
        logging::error("transcribe failed: " + err);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }
    logging::logf("INFO", "transcribed %zu segments", segments.size());

    std::wstring out = job.output.empty() ? derive_srt(job.input) : job.output;
    if (srt::write(segments, to_utf8(out), job.max_line_length, err)) {
        PostMessageW(hwnd, WM_APP_PROGRESS, 100, 0);
        logging::info("wrote " + to_utf8(out));
        post_str(hwnd, WM_APP_DONE, 1, L"Saved: " + out);
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
            post_str(hwnd, WM_APP_DONE, 1, L"Output folder not writable. Saved to Desktop: " + fb);
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
    job.translate = checked(g_translate);
    job.vad       = checked(g_vad);
    job.flash     = checked(g_flash);
    job.word_ts   = checked(g_wordts);
    job.center    = checked(g_center);
    job.isolate   = checked(g_isolate);
    job.max_line_length = checked(g_wrap) ? 42 : 0;
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

    // Checkbox row 1: transcription options.
    int x = m;
    struct { HWND h; int w; } r1[] = {
        {g_translate, 160}, {g_vad, 62}, {g_flash, 135}, {g_wordts, 150}, {g_wrap, 130}
    };
    for (auto& c : r1) { MoveWindow(c.h, x, y, c.w, 24, TRUE); x += c.w + 10; }
    y += 28;
    // Checkbox row 2: dialogue-audio options.
    x = m;
    struct { HWND h; int w; } r2[] = { {g_center, 180}, {g_isolate, 230} };
    for (auto& c : r2) { MoveWindow(c.h, x, y, c.w, 24, TRUE); x += c.w + 10; }
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

    g_translate = mk(hwnd, L"BUTTON", L"Translate → English", BS_AUTOCHECKBOX, ID_TRANSLATE);
    g_vad       = mk(hwnd, L"BUTTON", L"VAD",                 BS_AUTOCHECKBOX, ID_VAD);
    g_flash     = mk(hwnd, L"BUTTON", L"Flash attention",     BS_AUTOCHECKBOX, ID_FLASH);
    g_wordts    = mk(hwnd, L"BUTTON", L"Word timestamps",     BS_AUTOCHECKBOX, ID_WORDTS);
    g_wrap      = mk(hwnd, L"BUTTON", L"Wrap lines (42)",     BS_AUTOCHECKBOX, ID_WRAP);
    g_center    = mk(hwnd, L"BUTTON", L"Center channel (dialogue)", BS_AUTOCHECKBOX, ID_CENTER);
    g_isolate   = mk(hwnd, L"BUTTON", L"Isolate vocals (remove music)", BS_AUTOCHECKBOX, ID_ISOLATE);
    SendMessageW(g_vad,    BM_SETCHECK, BST_CHECKED, 0);
    SendMessageW(g_flash,  BM_SETCHECK, BST_CHECKED, 0);
    SendMessageW(g_wrap,   BM_SETCHECK, BST_CHECKED, 0);
    SendMessageW(g_center, BM_SETCHECK, BST_CHECKED, 0);

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
                    g_output_browse, g_translate, g_vad, g_flash, g_wordts, g_wrap, g_center,
                    g_isolate, g_model_lbl, g_model, g_lang_lbl, g_lang, g_vmodel_lbl, g_vmodel,
                    g_start, g_edit })
        SendMessageW(h, WM_SETFONT, (WPARAM)font, TRUE);
}

static void set_busy(bool busy) {
    for (HWND h : { g_start, g_input_browse, g_output_browse, g_input, g_output,
                    g_translate, g_vad, g_flash, g_wordts, g_wrap, g_center, g_isolate,
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
