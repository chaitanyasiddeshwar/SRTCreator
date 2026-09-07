#include "models.h"

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

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

void ensure_dir(const std::string& dir) {
    std::string cmd = "cmd /c if not exist \"" + dir + "\" mkdir \"" + dir + "\"";
    std::system(cmd.c_str());
}

// curl.exe ships with Windows 10/11. -L follows redirects, --fail errors on 4xx/5xx.
bool download_url(const std::string& url, const std::string& dest, std::string& err) {
    std::string tmp = dest + ".part";
    std::string cmd = "curl.exe -L --fail --progress-bar -o \"" + tmp + "\" \"" + url + "\"";
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::remove(tmp.c_str());
        err = "download failed (curl rc=" + std::to_string(rc) + "): " + url;
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
              std::string& out_path, std::string& err) {
    ensure_dir(models_dir);
    std::string fname = filename_for(name);
    out_path = models_dir + "\\" + fname;
    std::string url = std::string(kHfBase) + fname;
    std::fprintf(stderr, "[models] downloading %s\n", fname.c_str());
    return download_url(url, out_path, err);
}

bool resolve(const std::string& spec, const std::string& models_dir,
             bool allow_download, std::string& out_path, std::string& err) {
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
    return download(spec, models_dir, out_path, err);
}

std::string vad_model_path(const std::string& models_dir) {
    return models_dir + "\\" + kVadFile;
}

bool ensure_vad(const std::string& models_dir, bool allow_download,
                std::string& out_path, std::string& err) {
    out_path = vad_model_path(models_dir);
    if (file_exists(out_path)) return true;
    if (!allow_download) { err = "VAD model not present: " + out_path; return false; }
    ensure_dir(models_dir);
    std::string url = std::string(kVadBase) + kVadFile;
    std::fprintf(stderr, "[models] downloading VAD model %s\n", kVadFile);
    return download_url(url, out_path, err);
}

} // namespace models
