#include "log.h"

#include <windows.h>
#include <dbghelp.h>

#include <cstdarg>
#include <cstdio>
#include <mutex>

#pragma comment(lib, "dbghelp.lib")

namespace logging {

namespace {

FILE*       g_fp = nullptr;
std::mutex  g_mtx;
std::string g_tag = "app";
std::string g_path;

std::string local_appdata_dir() {
    const char* base = std::getenv("LOCALAPPDATA");
    std::string dir = (base && *base) ? std::string(base) + "\\SRTCreator" : std::string("logs");
    CreateDirectoryA(dir.c_str(), nullptr);
    return dir;
}

void write_line(const char* level, const char* msg) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!g_fp) return;
    SYSTEMTIME t; GetLocalTime(&t);
    std::fprintf(g_fp, "%04d-%02d-%02d %02d:%02d:%02d.%03d [%s] [%s] %s\n",
                 t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, t.wMilliseconds,
                 g_tag.c_str(), level, msg);
    std::fflush(g_fp);
}

void log_backtrace() {
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(proc, nullptr, TRUE);

    void* frames[64];
    USHORT n = CaptureStackBackTrace(0, 64, frames, nullptr);

    char symbuf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(symbuf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen   = 255;

    for (USHORT i = 0; i < n; ++i) {
        DWORD64 addr = (DWORD64)frames[i];

        HMODULE mod = nullptr;
        char modname[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)frames[i], &mod)) {
            char full[MAX_PATH];
            if (GetModuleFileNameA(mod, full, MAX_PATH)) {
                const char* b = strrchr(full, '\\');
                std::snprintf(modname, sizeof(modname), "%s", b ? b + 1 : full);
            }
        }
        DWORD64 base = (DWORD64)mod;

        DWORD64 disp = 0;
        char line[512];
        if (SymFromAddr(proc, addr, &disp, sym)) {
            IMAGEHLP_LINE64 il; il.SizeOfStruct = sizeof(il); DWORD ldisp = 0;
            if (SymGetLineFromAddr64(proc, addr, &ldisp, &il)) {
                const char* fb = strrchr(il.FileName, '\\');
                std::snprintf(line, sizeof(line), "  [%02u] %s!%s+0x%llx (%s:%lu)",
                              i, modname, sym->Name, (unsigned long long)disp,
                              fb ? fb + 1 : il.FileName, il.LineNumber);
            } else {
                std::snprintf(line, sizeof(line), "  [%02u] %s!%s+0x%llx",
                              i, modname, sym->Name, (unsigned long long)disp);
            }
        } else {
            std::snprintf(line, sizeof(line), "  [%02u] %s+0x%llx",
                          i, modname, (unsigned long long)(addr - base));
        }
        write_line("FATAL", line);
    }
}

LONG WINAPI seh_filter(EXCEPTION_POINTERS* ep) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "unhandled exception code=0x%08lx address=%p",
                  ep->ExceptionRecord->ExceptionCode,
                  ep->ExceptionRecord->ExceptionAddress);
    write_line("FATAL", buf);
    log_backtrace();
    return EXCEPTION_EXECUTE_HANDLER; // terminate after logging
}

void terminate_handler() {
    write_line("FATAL", "std::terminate (unhandled C++ exception)");
    if (auto e = std::current_exception()) {
        try { std::rethrow_exception(e); }
        catch (const std::exception& ex) { write_line("FATAL", (std::string("what(): ") + ex.what()).c_str()); }
        catch (...) { write_line("FATAL", "non-standard exception"); }
    }
    log_backtrace();
    std::abort();
}

} // namespace

void init(const char* app_tag) {
    if (g_fp) return;
    g_tag  = app_tag ? app_tag : "app";
    g_path = local_appdata_dir() + "\\srtcreator.log";
    g_fp   = std::fopen(g_path.c_str(), "a");

    char exe[MAX_PATH] = "?";
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    logf("INFO", "==== session start pid=%lu exe=%s ====", GetCurrentProcessId(), exe);

    SetUnhandledExceptionFilter(seh_filter);
    std::set_terminate(terminate_handler);
}

void info (const std::string& msg) { write_line("INFO",  msg.c_str()); }
void error(const std::string& msg) { write_line("ERROR", msg.c_str()); }

void logf(const char* level, const char* fmt, ...) {
    char buf[2048];
    va_list ap; va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    write_line(level, buf);
}

std::string log_path() { return g_path; }

} // namespace logging
