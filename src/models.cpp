#include "models.h"
#include "log.h"

#include <windows.h>
#include <urlmon.h>

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

#pragma comment(lib, "urlmon.lib")

namespace models {

namespace {

const char* kHfBase  = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
const char* kVadBase = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/";
const char* kVadFile = "ggml-silero-v5.1.2.bin";

bool file_exists(const std::string& p) {
    struct _stat64 st;
    return _stat64(p.c_str(), &st) == 0;
}

bool has_pathsep(const std::string& s) {
    return s.find('/') != std::string::npos || s.find('\\') != std::string::npos;
}

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::wstring to_wide(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}

// Create each component of a directory path (mkdir -p), no console spawned.
void ensure_dir(const std::string& dir) {
    std::string cur;
    for (size_t i = 0; i <= dir.size(); ++i) {
        if (i == dir.size() || dir[i] == '\\' || dir[i] == '/') {
            if (!cur.empty() && !(cur.size() == 2 && cur[1] == ':'))
                CreateDirectoryA(cur.c_str(), nullptr);
        }
        if (i < dir.size()) cur.push_back(dir[i]);
    }
}

// IBindStatusCallback that forwards URLDownloadToFile progress to a std::function.
class DownloadCallback : public IBindStatusCallback {
public:
    explicit DownloadCallback(const Progress& cb) : cb_(cb) {}

    // IUnknown (no real refcounting needed; lives on the stack for the call).
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (riid == IID_IUnknown || riid == IID_IBindStatusCallback) { *ppv = this; return S_OK; }
        *ppv = nullptr; return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef()  override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }

    HRESULT STDMETHODCALLTYPE OnStartBinding(DWORD, IBinding*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPriority(LONG*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE OnLowResource(DWORD) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE OnStopBinding(HRESULT, LPCWSTR) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetBindInfo(DWORD*, BINDINFO*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE OnDataAvailable(DWORD, DWORD, FORMATETC*, STGMEDIUM*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE OnObjectAvailable(REFIID, IUnknown*) override { return S_OK; }

    HRESULT STDMETHODCALLTYPE OnProgress(ULONG progress, ULONG progressMax,
                                         ULONG, LPCWSTR) override {
        if (cb_ && progressMax) {
            int pct = (int)((ULONGLONG)progress * 100 / progressMax);
            if (pct != last_) { last_ = pct; cb_(pct); }
        }
        return S_OK;
    }
private:
    Progress cb_;
    int      last_ = -1;
};

bool download_url(const std::string& url, const std::string& dest, std::string& err,
                  const Progress& on_progress) {
    std::string tmp = dest + ".part";
    DownloadCallback cb(on_progress);
    HRESULT hr = URLDownloadToFileW(nullptr, to_wide(url).c_str(), to_wide(tmp).c_str(),
                                    0, &cb);
    if (FAILED(hr)) {
        std::remove(tmp.c_str());
        char buf[64]; std::snprintf(buf, sizeof(buf), "0x%08lx", (unsigned long)hr);
        err = "download failed (" + std::string(buf) + "): " + url;
        return false;
    }
    std::remove(dest.c_str());
    if (std::rename(tmp.c_str(), dest.c_str()) != 0) {
        err = "could not finalize download: " + dest;
        return false;
    }
    return true;
}

} // namespace

std::string default_models_dir() {
    const char* local = std::getenv("LOCALAPPDATA");
    if (local && *local) return std::string(local) + "\\SRTCreator\\models";
    return "models";
}

std::string filename_for(const std::string& name) {
    return "ggml-" + name + ".bin";
}

bool download(const std::string& name, const std::string& models_dir,
              std::string& out_path, std::string& err, const Progress& on_progress) {
    ensure_dir(models_dir);
    std::string fname = filename_for(name);
    out_path = models_dir + "\\" + fname;
    std::string url = std::string(kHfBase) + fname;
    logging::logf("INFO", "downloading model %s", fname.c_str());
    return download_url(url, out_path, err, on_progress);
}

bool resolve(const std::string& spec, const std::string& models_dir,
             bool allow_download, std::string& out_path, std::string& err,
             const Progress& on_progress) {
    // Explicit path or filename.
    if (has_pathsep(spec) || ends_with(spec, ".bin") || ends_with(spec, ".gguf")) {
        if (file_exists(spec)) { out_path = spec; return true; }
        std::string cand = models_dir + "\\" + spec;
        if (file_exists(cand)) { out_path = cand; return true; }
        err = "model file not found: " + spec;
        return false;
    }
    // Model name.
    std::string cand = models_dir + "\\" + filename_for(spec);
    if (file_exists(cand)) { out_path = cand; return true; }
    if (!allow_download) {
        err = "model '" + spec + "' not present (" + cand +
              "); run once with network access or: srt --download " + spec;
        return false;
    }
    return download(spec, models_dir, out_path, err, on_progress);
}

std::string vad_model_path(const std::string& models_dir) {
    return models_dir + "\\" + kVadFile;
}

bool ensure_vad(const std::string& models_dir, bool allow_download,
                std::string& out_path, std::string& err, const Progress& on_progress) {
    out_path = vad_model_path(models_dir);
    if (file_exists(out_path)) return true;
    if (!allow_download) { err = "VAD model not present: " + out_path; return false; }
    ensure_dir(models_dir);
    std::string url = std::string(kVadBase) + kVadFile;
    logging::logf("INFO", "downloading VAD model %s", kVadFile);
    return download_url(url, out_path, err, on_progress);
}

} // namespace models
