#include "separate.h"
#include "log.h"
#include "models.h"

#ifndef NOMINMAX
#define NOMINMAX   // keep windows.h from defining max()/min() macros (breaks pocketfft/std)
#endif
#include <windows.h>
#include "pocketfft_hdronly.h"
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <array>
#include <cmath>
#include <complex>
#include <sys/stat.h>

namespace separate {

namespace {

constexpr int   HOP = 1024;
constexpr float PI  = 3.14159265358979323846f;

// torch.hann_window(n) with periodic=True: 0.5 - 0.5*cos(2*pi*i/n).
std::vector<float> hann(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) w[i] = 0.5f - 0.5f * std::cos(2.0f * PI * i / n);
    return w;
}

std::wstring to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}

// STFT of one chunk (chunk_len samples) with center padding (zeros; the padded
// region is trimmed away downstream). Produces dim_t frames of n_bins bins;
// only the first dim_f bins are written into `out` as interleaved (re,im),
// laid out as out[(k*dim_f + f)*dim_t + t] for k in {0=re,1=im}.
void stft_chunk(const float* chunk, int chunk_len, const std::vector<float>& win,
                int n_fft, int dim_f, int dim_t,
                float* out_re, float* out_im) {
    int pad = n_fft / 2;
    int n_bins = n_fft / 2 + 1;
    std::vector<float> padded((size_t)chunk_len + 2 * pad, 0.0f);
    for (int i = 0; i < chunk_len; ++i) padded[pad + i] = chunk[i];

    std::vector<float> frame(n_fft);
    std::vector<std::complex<float>> spec(n_bins);
    pocketfft::shape_t shape{ (size_t)n_fft };
    pocketfft::stride_t stin{ (ptrdiff_t)sizeof(float) };
    pocketfft::stride_t stout{ (ptrdiff_t)sizeof(std::complex<float>) };

    for (int t = 0; t < dim_t; ++t) {
        int start = t * HOP;
        for (int i = 0; i < n_fft; ++i) frame[i] = padded[start + i] * win[i];
        pocketfft::r2c(shape, stin, stout, 0, pocketfft::FORWARD,
                       frame.data(), spec.data(), 1.0f);
        for (int f = 0; f < dim_f; ++f) {
            out_re[(size_t)f * dim_t + t] = spec[f].real();
            out_im[(size_t)f * dim_t + t] = spec[f].imag();
        }
    }
}

// iSTFT: dim_f/dim_t (re,im) spectrogram -> chunk_len samples (center removed).
void istft_chunk(const float* in_re, const float* in_im,
                 const std::vector<float>& win, int n_fft, int dim_f, int dim_t,
                 int chunk_len, std::vector<float>& out) {
    int pad = n_fft / 2;
    int n_bins = n_fft / 2 + 1;
    size_t padded_len = (size_t)chunk_len + 2 * pad;
    std::vector<float> acc(padded_len, 0.0f), norm(padded_len, 0.0f);

    std::vector<std::complex<float>> spec(n_bins);
    std::vector<float> frame(n_fft);
    pocketfft::shape_t shape{ (size_t)n_fft };
    pocketfft::stride_t stin{ (ptrdiff_t)sizeof(std::complex<float>) };
    pocketfft::stride_t stout{ (ptrdiff_t)sizeof(float) };

    for (int t = 0; t < dim_t; ++t) {
        for (int f = 0; f < n_bins; ++f) {
            if (f < dim_f) spec[f] = { in_re[(size_t)f * dim_t + t], in_im[(size_t)f * dim_t + t] };
            else           spec[f] = { 0.0f, 0.0f };
        }
        pocketfft::c2r(shape, stin, stout, 0, pocketfft::BACKWARD,
                       spec.data(), frame.data(), 1.0f / n_fft);
        int start = t * HOP;
        for (int i = 0; i < n_fft; ++i) {
            acc[start + i]  += frame[i] * win[i];
            norm[start + i] += win[i] * win[i];
        }
    }
    out.resize(chunk_len);
    for (int i = 0; i < chunk_len; ++i) {
        float n = norm[pad + i];
        out[i] = n > 1e-8f ? acc[pad + i] / n : 0.0f;
    }
}

bool file_exists(const std::string& p) { struct _stat64 st; return _stat64(p.c_str(), &st) == 0; }

} // namespace

// UVR model repo (public mirror) + params from UVR's model_data.json.
static const char* REPO = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/";
static const std::vector<ModelInfo> g_registry = {
    { "Kim_Vocal_2",           "Kim_Vocal_2.onnx",
      "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx",
      { 7680, 3072, 256, 1.009f, true } },
    { "UVR-MDX-NET-Inst_HQ_3", "UVR-MDX-NET-Inst_HQ_3.onnx",
      "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx",
      { 6144, 3072, 256, 1.022f, false } },
    { "UVR_MDXNET_KARA_2",     "UVR_MDXNET_KARA_2.onnx",
      "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_KARA_2.onnx",
      { 5120, 2048, 256, 1.065f, false } },
};

const std::vector<ModelInfo>& registry() { return g_registry; }

const ModelInfo* find(const std::string& name) {
    for (const auto& m : g_registry) if (name == m.name) return &m;
    return nullptr;
}

bool ensure_model(const std::string& name, const std::string& models_dir,
                  bool allow_download, std::string& out_path, Params& params,
                  std::string& err, const std::function<void(int)>& on_progress) {
    const ModelInfo* mi = find(name);
    if (!mi) { err = "unknown vocal model: " + name; return false; }
    params = mi->params;
    out_path = models_dir + "\\" + mi->filename;
    if (file_exists(out_path)) return true;
    if (!allow_download) { err = "vocal model not present: " + out_path; return false; }
    logging::logf("INFO", "downloading vocal model %s", mi->filename);
    return models::fetch(mi->url, out_path, err, on_progress);
}

bool isolate_vocals(const std::vector<float>& stereo, const std::string& model_path,
                    const Params& p, std::vector<float>& out_mono,
                    const std::function<void(int)>& on_progress, std::string& err) {
    out_mono.clear();
    const int   n_fft   = p.n_fft;
    const int   dim_f   = p.dim_f;
    const int   dim_t   = p.dim_t;
    const int   trim    = n_fft / 2;
    const int   chunk   = HOP * (dim_t - 1);   // samples per model chunk
    const int   gen     = chunk - 2 * trim;    // usable samples after trimming overlap
    if (gen <= 0) { err = "invalid MDX params"; return false; }

    const std::vector<float> win = hann(n_fft);
    const size_t total = stereo.size() / 2;    // frames (per channel)

    // ---- ONNX Runtime session on DirectML ----
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "srt-sep");
        Ort::SessionOptions so;
        so.DisableMemPattern();
        so.SetExecutionMode(ORT_SEQUENTIAL);
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
        Ort::Session session(env, to_wide(model_path).c_str(), so);

        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name  = session.GetInputNameAllocated(0, alloc);
        auto out_name = session.GetOutputNameAllocated(0, alloc);
        const char* in_names[]  = { in_name.get() };
        const char* out_names[] = { out_name.get() };
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> shape{ 1, 4, dim_f, dim_t };
        const size_t plane = (size_t)dim_f * dim_t;

        std::vector<float> input((size_t)4 * plane);
        std::vector<float> ch0(chunk), ch1(chunk), rec0, rec1;
        out_mono.reserve(total);

        // Front pad by trim, then step by gen; each processed window is `chunk`
        // long, reconstructed, and trimmed by `trim` on each side.
        long long pos = -trim;
        size_t nchunks = (total + gen - 1) / gen + 1;
        size_t done = 0;
        while (pos < (long long)total) {
            for (int i = 0; i < chunk; ++i) {
                long long s = pos + i;
                float a = 0, b = 0;
                if (s >= 0 && s < (long long)total) { a = stereo[(size_t)s * 2]; b = stereo[(size_t)s * 2 + 1]; }
                ch0[i] = a; ch1[i] = b;
            }
            // STFT both channels into the 4-plane input (k: 0=c0re,1=c0im,2=c1re,3=c1im).
            stft_chunk(ch0.data(), chunk, win, n_fft, dim_f, dim_t,
                       input.data() + 0 * plane, input.data() + 1 * plane);
            stft_chunk(ch1.data(), chunk, win, n_fft, dim_f, dim_t,
                       input.data() + 2 * plane, input.data() + 3 * plane);

            Ort::Value in = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                            shape.data(), shape.size());
            auto outs = session.Run(Ort::RunOptions{ nullptr }, in_names, &in, 1, out_names, 1);
            float* od = outs[0].GetTensorMutableData<float>();

            istft_chunk(od + 0 * plane, od + 1 * plane, win, n_fft, dim_f, dim_t, chunk, rec0);
            istft_chunk(od + 2 * plane, od + 3 * plane, win, n_fft, dim_f, dim_t, chunk, rec1);

            // Trim the overlap and append the mono average (vocals).
            for (int i = trim; i < chunk - trim; ++i) {
                long long gpos = pos + i; // original-timeline sample index
                if (gpos < 0 || gpos >= (long long)total) continue;
                float v0 = rec0[i], v1 = rec1[i];
                if (!p.vocals_primary) { // instrumental model: vocals = mix - out
                    v0 = stereo[(size_t)gpos * 2]     - v0;
                    v1 = stereo[(size_t)gpos * 2 + 1] - v1;
                }
                out_mono.push_back(0.5f * (v0 + v1) * p.compensate);
            }

            pos += gen;
            if (on_progress) { int pct = (int)(++done * 100 / (nchunks ? nchunks : 1)); on_progress(pct > 100 ? 100 : pct); }
        }
    } catch (const Ort::Exception& e) {
        err = std::string("onnxruntime: ") + e.what();
        return false;
    } catch (const std::exception& e) {
        err = std::string("separation failed: ") + e.what();
        return false;
    }

    if (out_mono.empty()) { err = "no vocals produced"; return false; }
    return true;
}

} // namespace separate
