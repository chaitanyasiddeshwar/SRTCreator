#pragma once
#include <functional>
#include <string>
#include <vector>

// Vocal isolation with an MDX-Net ONNX model (UVR-style), run on the GPU via
// ONNX Runtime's DirectML backend. Removing music/effects before whisper hugely
// improves dialogue transcription over scored scenes.
namespace separate {

// Per-model MDX-Net parameters (from UVR's model_data.json).
struct Params {
    int   n_fft;          // e.g. 7680 (Kim_Vocal_2)
    int   dim_f;          // kept frequency bins, e.g. 3072
    int   dim_t;          // frames per chunk, e.g. 256
    float compensate;     // output gain, e.g. 1.009
    bool  vocals_primary; // true: model outputs Vocals; false: outputs Instrumental (vocals = mix - out)
};

// A selectable vocal-isolation model.
struct ModelInfo {
    const char* name;      // short id, e.g. "Kim_Vocal_2"
    const char* filename;  // ggml-less file, e.g. "Kim_Vocal_2.onnx"
    const char* url;       // download source
    Params      params;
};

// Registry of known models (first entry is the recommended default).
const std::vector<ModelInfo>& registry();
const ModelInfo* find(const std::string& name);

// Resolve a model by name to a local .onnx path, downloading if needed.
bool ensure_model(const std::string& name, const std::string& models_dir,
                  bool allow_download, std::string& out_path, Params& params,
                  std::string& err, const std::function<void(int)>& on_progress = {});

// Isolate vocals from interleaved stereo audio at 44100 Hz. Returns the vocal
// stem as mono float at 44100 Hz. on_progress reports 0-100. Returns false/err
// on failure (caller can fall back to the raw mix).
bool isolate_vocals(const std::vector<float>& stereo_44100,
                    const std::string& onnx_model_path,
                    const Params& params,
                    std::vector<float>& out_mono_44100,
                    const std::function<void(int)>& on_progress,
                    std::string& err);

} // namespace separate
