#pragma once
#include <string>

namespace models {

// Default model store: %LOCALAPPDATA%\SRTCreator\models (or "models" fallback).
std::string default_models_dir();

// ggml filename for a model name, e.g. "large-v3-turbo-q8_0" ->
// "ggml-large-v3-turbo-q8_0.bin".
std::string filename_for(const std::string& name);

// Resolve a model spec (a name, a bare filename, or a full path) to a local
// file. If it is a name and not present, download it when allow_download.
bool resolve(const std::string& spec, const std::string& models_dir,
             bool allow_download, std::string& out_path, std::string& err);

// Download a model by name into models_dir.
bool download(const std::string& name, const std::string& models_dir,
              std::string& out_path, std::string& err);

// Silero VAD model path + fetch.
std::string vad_model_path(const std::string& models_dir);
bool ensure_vad(const std::string& models_dir, bool allow_download,
                std::string& out_path, std::string& err);

} // namespace models
