// SRTCreator GUI - a minimal drag-and-drop window.
//
// Drop a media file onto the window: it decodes, transcribes on the GPU, shows
// the subtitles live in the panel with a progress bar, and writes <input>.srt.
// All CLI toggles are exposed as checkboxes / dropdowns.

#ifndef UNICODE
#define UNICODE
#endif
#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "audio.h"
#include "transcribe.h"
#include "srt.h"
#include "models.h"

// Enable visual styles (themed common controls v6).
#pragma comment(linker, "\"/manifestdependency:type='win32' \
name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#pragma comment(lib, "comctl32.lib")

// ---- worker -> UI messages (marshaled onto the UI thread) ----
#define WM_APP_SEGMENT  (WM_APP + 1) // lParam = wchar_t* (heap): append line
#define WM_APP_PROGRESS (WM_APP + 2) // wParam = 0..100
#define WM_APP_STATUS   (WM_APP + 3) // lParam = wchar_t* (heap)
#define WM_APP_MARQUEE  (WM_APP + 4) // wParam = 1 start / 0 stop
#define WM_APP_DONE     (WM_APP + 5) // lParam = wchar_t* (heap); wParam = success

// ---- control IDs ----
enum {
    ID_TRANSLATE = 1001, ID_VAD, ID_FLASH, ID_WORDTS, ID_WRAP,
    ID_MODEL, ID_LANG, ID_EDIT, ID_PROGRESS, ID_STATUS
};

static HWND g_status, g_translate, g_vad, g_flash, g_wordts, g_wrap;
static HWND g_model, g_lang, g_edit, g_progress;
static std::atomic<bool> g_running{false};

// ---- utf8 <-> wide ----
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

static std::wstring combo_text(HWND h) {
    wchar_t buf[128] = {0};
    GetWindowTextW(h, buf, 128);
    return buf;
}

static void append_line(HWND hEdit, const wchar_t* text) {
    int len = GetWindowTextLengthW(hEdit);
    SendMessageW(hEdit, EM_SETSEL, len, len);
    SendMessageW(hEdit, EM_REPLACESEL, FALSE, (LPARAM)text);
}

// ---- the transcription job (runs on a worker thread) ----
struct Job {
    HWND hwnd;
    std::wstring input;
    std::string model, language;
    bool translate, vad, flash, word_ts;
    int max_line_length;
};

static void run_job(Job job) {
    HWND hwnd = job.hwnd;
    std::string input = to_utf8(job.input);
    std::string err;

    std::string models_dir = models::default_models_dir();

    post_str(hwnd, WM_APP_STATUS, 0, L"Preparing model (first run may download)…");
    PostMessageW(hwnd, WM_APP_MARQUEE, 1, 0);

    std::string model_path;
    if (!models::resolve(job.model, models_dir, true, model_path, err)) {
        PostMessageW(hwnd, WM_APP_MARQUEE, 0, 0);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }

    std::string vad_path;
    bool vad = job.vad;
    if (vad && !models::ensure_vad(models_dir, true, vad_path, err)) {
        vad = false; // continue without VAD
    }

    post_str(hwnd, WM_APP_STATUS, 0, L"Decoding audio…");
    std::vector<float> pcm;
    audio::DecodeOptions dopts;
    if (!audio::decode(input, dopts, pcm, err)) {
        PostMessageW(hwnd, WM_APP_MARQUEE, 0, 0);
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }

    PostMessageW(hwnd, WM_APP_MARQUEE, 0, 0);
    PostMessageW(hwnd, WM_APP_PROGRESS, 0, 0);
    post_str(hwnd, WM_APP_STATUS, 0, L"Transcribing…");

    std::vector<srt::Segment> segments;
    transcribe::Options t;
    t.model_path      = model_path;
    t.language        = job.language;
    t.translate       = job.translate;
    t.flash_attn      = job.flash;
    t.vad             = vad;
    t.vad_model_path  = vad_path;
    t.word_timestamps = job.word_ts;
    // Display only - transcribe::run fills `segments` itself (the file's source
    // of truth); pushing here too would duplicate every line.
    t.on_progress = [hwnd](int pct) { PostMessageW(hwnd, WM_APP_PROGRESS, (WPARAM)pct, 0); };
    t.on_segment  = [hwnd](const srt::Segment& s) {
        std::wstring line = L"[" + to_wide(srt::format_timestamp(s.t0)).substr(0, 8) + L"]  "
                          + to_wide(s.text) + L"\r\n";
        post_str(hwnd, WM_APP_SEGMENT, 0, line);
    };

    if (!transcribe::run(pcm, t, segments, err)) {
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }

    // Output path: <input>.srt (extension replaced).
    std::wstring out = job.input;
    size_t slash = out.find_last_of(L"/\\");
    size_t dot   = out.find_last_of(L'.');
    out = (dot == std::wstring::npos || (slash != std::wstring::npos && dot < slash))
              ? out + L".srt" : out.substr(0, dot) + L".srt";

    if (!srt::write(segments, to_utf8(out), job.max_line_length, err)) {
        post_str(hwnd, WM_APP_DONE, 0, L"Error: " + to_wide(err));
        return;
    }
    PostMessageW(hwnd, WM_APP_PROGRESS, 100, 0);
    post_str(hwnd, WM_APP_DONE, 1, L"Saved: " + out);
}

static void start_job(HWND hwnd, const std::wstring& path) {
    if (g_running.exchange(true)) return; // one at a time
    SetWindowTextW(g_edit, L"");

    Job job;
    job.hwnd            = hwnd;
    job.input           = path;
    job.model           = to_utf8(combo_text(g_model));
    job.language        = to_utf8(combo_text(g_lang));
    job.translate       = checked(g_translate);
    job.vad             = checked(g_vad);
    job.flash           = checked(g_flash);
    job.word_ts         = checked(g_wordts);
    job.max_line_length = checked(g_wrap) ? 42 : 0;

    std::thread(run_job, job).detach();
}

// ---- layout ----
static void layout(HWND hwnd) {
    RECT rc; GetClientRect(hwnd, &rc);
    int W = rc.right, H = rc.bottom, m = 10;
    MoveWindow(g_status, m, m, W - 2 * m, 20, TRUE);

    int y = m + 26, x = m;
    struct { HWND h; int w; } cbs[] = {
        {g_translate, 150}, {g_vad, 70}, {g_flash, 110}, {g_wordts, 150}, {g_wrap, 130}
    };
    for (auto& c : cbs) { MoveWindow(c.h, x, y, c.w, 22, TRUE); x += c.w; }

    y += 28;
    MoveWindow(g_model, m + 50, y, 220, 200, TRUE);
    MoveWindow(g_lang,  m + 50 + 220 + 60, y, 120, 200, TRUE);

    int top = y + 32;
    int prog_h = 20;
    MoveWindow(g_edit, m, top, W - 2 * m, H - top - prog_h - 2 * m, TRUE);
    MoveWindow(g_progress, m, H - prog_h - m, W - 2 * m, prog_h, TRUE);
}

static HWND mk(HWND parent, const wchar_t* cls, const wchar_t* text, DWORD style, int id) {
    return CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style,
                           0, 0, 0, 0, parent, (HMENU)(INT_PTR)id,
                           (HINSTANCE)GetWindowLongPtrW(parent, GWLP_HINSTANCE), nullptr);
}

static void create_controls(HWND hwnd) {
    g_status = mk(hwnd, L"STATIC", L"Drop a movie or audio file here to create subtitles.", SS_LEFT, ID_STATUS);

    g_translate = mk(hwnd, L"BUTTON", L"Translate → English", BS_AUTOCHECKBOX, ID_TRANSLATE);
    g_vad       = mk(hwnd, L"BUTTON", L"VAD",              BS_AUTOCHECKBOX, ID_VAD);
    g_flash     = mk(hwnd, L"BUTTON", L"Flash attention",  BS_AUTOCHECKBOX, ID_FLASH);
    g_wordts    = mk(hwnd, L"BUTTON", L"Word timestamps",  BS_AUTOCHECKBOX, ID_WORDTS);
    g_wrap      = mk(hwnd, L"BUTTON", L"Wrap lines (42)",  BS_AUTOCHECKBOX, ID_WRAP);
    SendMessageW(g_vad,   BM_SETCHECK, BST_CHECKED, 0);
    SendMessageW(g_flash, BM_SETCHECK, BST_CHECKED, 0);
    SendMessageW(g_wrap,  BM_SETCHECK, BST_CHECKED, 0);

    mk(hwnd, L"STATIC", L"Model:", SS_LEFT, 0);
    g_model = mk(hwnd, L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_VSCROLL, ID_MODEL);
    const wchar_t* models_[] = { L"large-v3-turbo-q8_0", L"large-v3-turbo", L"large-v3",
                                 L"medium", L"small", L"base", L"tiny",
                                 L"small.en", L"base.en", L"tiny.en" };
    for (auto s : models_) SendMessageW(g_model, CB_ADDSTRING, 0, (LPARAM)s);
    SendMessageW(g_model, CB_SETCURSEL, 0, 0);

    mk(hwnd, L"STATIC", L"Lang:", SS_LEFT, 0);
    g_lang = mk(hwnd, L"COMBOBOX", L"", CBS_DROPDOWN | WS_VSCROLL, ID_LANG);
    const wchar_t* langs[] = { L"auto", L"en", L"es", L"fr", L"de", L"it", L"pt",
                               L"nl", L"ru", L"zh", L"ja", L"ko", L"hi", L"ar" };
    for (auto s : langs) SendMessageW(g_lang, CB_ADDSTRING, 0, (LPARAM)s);
    SetWindowTextW(g_lang, L"auto");

    g_edit = mk(hwnd, L"EDIT", L"",
                ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_BORDER, ID_EDIT);
    g_progress = mk(hwnd, PROGRESS_CLASSW, L"", 0, ID_PROGRESS);
    SendMessageW(g_progress, PBM_SETRANGE, 0, MAKELPARAM(0, 100));

    // A slightly larger UI font for readability.
    HFONT font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    for (HWND h : { g_status, g_translate, g_vad, g_flash, g_wordts, g_wrap, g_model, g_lang, g_edit })
        SendMessageW(h, WM_SETFONT, (WPARAM)font, TRUE);
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_CREATE:
        create_controls(hwnd);
        DragAcceptFiles(hwnd, TRUE);
        return 0;
    case WM_SIZE:
        layout(hwnd);
        return 0;
    case WM_GETMINMAXINFO: {
        MINMAXINFO* mmi = (MINMAXINFO*)lp;
        mmi->ptMinTrackSize.x = 640; mmi->ptMinTrackSize.y = 400;
        return 0;
    }
    case WM_DROPFILES: {
        if (!g_running) {
            HDROP drop = (HDROP)wp;
            wchar_t path[MAX_PATH];
            if (DragQueryFileW(drop, 0, path, MAX_PATH)) start_job(hwnd, path);
        }
        DragFinish((HDROP)wp);
        return 0;
    }
    case WM_APP_SEGMENT: {
        wchar_t* s = (wchar_t*)lp;
        append_line(g_edit, s);
        free(s);
        return 0;
    }
    case WM_APP_PROGRESS:
        SendMessageW(g_progress, PBM_SETPOS, wp, 0);
        return 0;
    case WM_APP_STATUS: {
        wchar_t* s = (wchar_t*)lp;
        SetWindowTextW(g_status, s);
        free(s);
        return 0;
    }
    case WM_APP_MARQUEE: {
        LONG_PTR st = GetWindowLongPtrW(g_progress, GWL_STYLE);
        if (wp) { SetWindowLongPtrW(g_progress, GWL_STYLE, st | PBS_MARQUEE);
                  SendMessageW(g_progress, PBM_SETMARQUEE, TRUE, 30); }
        else    { SendMessageW(g_progress, PBM_SETMARQUEE, FALSE, 0);
                  SetWindowLongPtrW(g_progress, GWL_STYLE, st & ~PBS_MARQUEE); }
        return 0;
    }
    case WM_APP_DONE: {
        wchar_t* s = (wchar_t*)lp;
        SetWindowTextW(g_status, s);
        free(s);
        g_running = false;
        return 0;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int nShow) {
    SetProcessDPIAware();
    INITCOMMONCONTROLSEX icc{ sizeof(icc), ICC_PROGRESS_CLASS | ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&icc);

    WNDCLASSW wc{};
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hInst;
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = L"SRTCreatorWindow";
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(WS_EX_ACCEPTFILES, wc.lpszClassName, L"SRTCreator",
                                WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                                900, 620, nullptr, nullptr, hInst, nullptr);
    ShowWindow(hwnd, nShow);
    UpdateWindow(hwnd);

    // Optional: a file passed on the command line (e.g. "Open with…") starts
    // immediately, using the current default toggle states.
    {
        int argc = 0;
        LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
        if (argv) {
            if (argc >= 2 && GetFileAttributesW(argv[1]) != INVALID_FILE_ATTRIBUTES)
                start_job(hwnd, argv[1]);
            LocalFree(argv);
        }
    }

    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0)) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    return 0;
}
