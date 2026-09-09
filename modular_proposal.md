# Modular / Productisation Refactor Proposal

**Status:** exploration + plan for discussion (branch `module_refactor`). Nothing here is implemented yet.

**Goal (as stated):** stop *building* the heavy third-party dependencies as part of our
build. Consume them as **prebuilt runtime DLLs**, leaving the main executables (`srt.exe`
CLI, `srtgui.exe` GUI) as a thin orchestration layer over our own code. Make CUDA
**optional** — detect it at runtime, ask the user to install the right pieces if they want
GPU, and **fall back to CPU** when it is unavailable. End result: anyone can install and
run this on their machine by grabbing the right (small) set of dependencies.

---

## 1. Where we are today (measured)

The build compiles **whisper.cpp + ggml + the CUDA backend from source** and static-links
them into each exe. FFmpeg and ONNX Runtime are already consumed as prebuilt shared DLLs.

| Dependency | How it's consumed now | Built by us? | Runtime form |
|---|---|---|---|
| **whisper.cpp + ggml + ggml-cuda** | vendored submodule, `add_subdirectory`, `GGML_CUDA=ON`, `BUILD_SHARED_LIBS=OFF` | **YES (needs CUDA Toolkit 13 + nvcc + Ninja)** | static, inside the 67 MB exe |
| **CUDA runtime** (cudart, cublas, cublasLt) | copied from `%CUDA_PATH%` next to exe | no (bundled) | DLLs shipped |
| **FFmpeg** (avformat/avcodec/avutil/swresample) | BtbN LGPL prebuilt shared, `third_party/ffmpeg` | no | DLLs shipped ✅ |
| **ONNX Runtime + DirectML** | prebuilt, `third_party/onnxruntime` | no | DLLs shipped ✅ |

### Current distribution footprint (from `build\`)

```
srt.exe / srtgui.exe        67 MB each   (static whisper+ggml+CUDA kernels for SM 8.6)
cublasLt64_13.dll          478 MB   ← the elephant
cublas64_13.dll             50 MB
cudart64_13.dll            0.5 MB
avcodec-63.dll              71 MB
avformat/avfilter/...      ~60 MB   (FFmpeg set)
onnxruntime.dll             17 MB
DirectML.dll                18 MB
--------------------------------------------------
TOTAL                     ~760 MB, dominated by CUDA (cublasLt)
```

### Two problems this creates

1. **Build friction.** Anyone building the project needs the full CUDA Toolkit (nvcc),
   a matching MSVC, and the Ninja-vs-VS-generator dance (CLAUDE.md §4.1). The 67 MB exe is
   ~SM-8.6 CUDA kernels baked in — non-portable across GPU generations unless we compile
   for more archs (even bigger).
2. **Distribution friction.** ~760 MB, ~530 MB of which is CUDA, and it only runs on an
   NVIDIA GPU with a compatible driver. No CPU-only path, no AMD/Intel path.

**Important:** the isolation path is *already* productised and vendor-neutral — ONNX Runtime
with the **DirectML** execution provider runs on any DirectX-12 GPU (NVIDIA/AMD/Intel), no
CUDA. So **the ONLY hard CUDA dependency in the whole project is whisper transcription.**
That single fact is what makes this refactor tractable.

---

## 2. Key findings

### 2.1 Upstream ships prebuilt whisper Windows DLLs

`ggml-org/whisper.cpp` GitHub Releases attach ready-to-run Windows x64 binaries
(verified via the GitHub API):

| Asset | Backend | Size | Bundles CUDA runtime? |
|---|---|---|---|
| `whisper-bin-x64.zip` | CPU | **8 MB** | n/a |
| `whisper-blas-bin-x64.zip` | CPU + OpenBLAS | 20 MB | n/a |
| `whisper-cublas-11.8.0-bin-x64.zip` | CUDA 11.8 | **257 MB** | yes |
| `whisper-cublas-12.4.0-bin-x64.zip` | CUDA 12.4 | **640 MB** | yes (cublasLt included) |

Each zip contains `whisper.dll`, `ggml*.dll`, headers, and the `whisper.lib` import lib —
i.e. exactly the shape we already consume FFmpeg/ONNX in. The cuBLAS zips **bundle the CUDA
runtime DLLs**, so an end user needs only an NVIDIA **driver**, not the CUDA Toolkit.

So consuming prebuilt whisper is entirely feasible; the CPU build is tiny (8 MB).

### 2.2 `GGML_BACKEND_DL` = runtime backend selection + automatic CPU fallback

ggml supports **dynamic backend loading** (`GGML_BACKEND_DL=ON`, upstreamed in llama.cpp
PR #10469, present in ggml/whisper). When enabled:

- Each backend is a **separate DLL**: `ggml-cpu.dll` (+ CPU-feature variants
  `ggml-cpu-haswell.dll`, `-icelake`, etc.), `ggml-cuda.dll`, `ggml-vulkan.dll`.
- The app calls `ggml_backend_load_all()` at startup; ggml scans the exe directory, **loads
  whatever backend DLLs are present**, scores CPU variants against the host CPU, and if no
  GPU backend loads it uses CPU as the fallback.
- This is *exactly* the "CUDA if available, else CPU" model — with **no code branches** on
  our side beyond querying the selected device and reporting it.

Consequence: we can ship a CPU base and make GPU an **additive** drop-in — put
`ggml-cuda.dll` (+ CUDA runtime) or `ggml-vulkan.dll` next to the exe and it "lights up".

### 2.3 Vulkan sidesteps the 478 MB cublasLt problem

The CUDA backend links cuBLAS/cuBLASLt; `cublasLt64_*.dll` is ~478 MB because it carries
GEMM kernels for many architectures. The **Vulkan** ggml backend needs none of that — it
compiles compute shaders and runs on the GPU vendor's **Vulkan driver** (already installed
with any modern NVIDIA/AMD/Intel driver). Reported cost vs CUDA is ~5–15% slower
(consistent with CLAUDE.md's original note), for a runtime footprint of a single small
`ggml-vulkan.dll` and **zero CUDA files**. Upstream does not attach a Windows Vulkan zip in
every release, so this backend would need a one-time build (see §5), but it is the single
biggest lever for a small, universal GPU package.

### 2.4 Segment mode being the default lets us drop the vendored-source patch

`patches/whisper.patch` modifies whisper's **VAD-timeline token remapping** (the DTW timing
fix, CLAUDE.md §14). It only matters when **whisper's internal VAD** is active — i.e. the
**legacy multi-pass** path. The new default **segment mode transcribes each speech region as
an isolated slice with whisper's VAD OFF**, so the patched code path is never exercised;
`whisper_full_get_token_t0/t1` return correct slice-local times from a stock build.

**All whisper symbols we call are stock API** (verified: `whisper_full`, `..._get_token_t0/t1`,
`..._n_vad_segments`, the standalone `whisper_vad_*` used by segment mode). So:

- **Segment mode works correctly against stock prebuilt whisper DLLs** — no patch, no
  vendored source build.
- The only casualty is **multi-pass timing accuracy** (legacy path), which would regress to
  stock behaviour without the patch. Given multi-pass is now opt-in legacy, that is an
  acceptable trade — or we keep the patch and a from-source build as an *optional* dev path.

This is the linchpin: **the segment-mode-first direction is what makes a prebuilt,
CUDA-free, small distribution realistic.**

---

## 3. Proposed architecture

### 3.1 Thin exe + import libs (mirror the FFmpeg/ONNX pattern)

Replace `add_subdirectory(third_party/whisper.cpp)` + the CUDA build with **consuming
prebuilt whisper**, exactly like FFmpeg today: headers + `whisper.lib` import lib under
`third_party/whisper/{include,lib,bin}`. Then:

- `srtcore` links `whisper.lib` (import), `avformat/...`, `onnxruntime` — all import libs.
- The exe drops from 67 MB to ~1–3 MB (our code only). All heavy code lives in DLLs.
- **The build no longer needs the CUDA Toolkit or nvcc.** MSVC alone builds the exe; the
  Ninja-vs-VS-generator gotcha (§4.1) disappears (nothing CUDA to compile).

### 3.2 Runtime backend detection

At startup (`transcribe::Session::init`):

1. Call `ggml_backend_load_all()` (loads any backend DLLs next to the exe).
2. Query the registered devices; pick the best available (CUDA/Vulkan GPU, else CPU).
3. Log/report the chosen backend to CLI stderr and the GUI status bar
   ("GPU: NVIDIA (CUDA)" / "GPU: Vulkan" / "CPU (no GPU backend found)").
4. If a user *expected* GPU but got CPU, print a one-line hint pointing at the GPU pack /
   driver update.

Requires whisper/ggml built with `GGML_BACKEND_DL=ON` (stock upstream cuBLAS/CPU zips may
**not** be — see Open Questions §7). This is the main reason §5 includes a controlled build.

### 3.3 Tiered distribution

```
SRTCreator-base.zip           (~90 MB, runs on ANY Windows 10/11 x64)
  srt.exe, srtgui.exe         (~2 MB)
  whisper.dll, ggml-base.dll, ggml-cpu*.dll        (CPU transcription)
  avcodec/avformat/avutil/swresample-*.dll         (FFmpeg, decode any file)
  onnxruntime.dll, DirectML.dll                    (vocal isolation, any DX12 GPU)
  models/ (optional; or downloaded on first run)

+ GPU pack (pick one, drop next to the exe or via first-run downloader):
  SRTCreator-vulkan.zip       (~a few MB)   ggml-vulkan.dll   → any modern GPU, driver only
  SRTCreator-cuda.zip         (~500 MB)     ggml-cuda.dll + cudart/cublas/cublasLt
                                            → NVIDIA, maximum speed
```

- **Base is CPU + DirectML isolation** — universal, no GPU story required to run.
- **GPU is additive**: the app detects the dropped-in backend and uses it.
- **First-run experience (GUI):** detect an NVIDIA driver → offer the CUDA pack; else offer
  Vulkan; else stay on CPU. Downloads go into the exe directory (same mechanism as our
  in-process model downloads today).

### 3.4 CUDA-as-user-install variants (for discussion)

Options for the CUDA case, cheapest-for-us to most-convenient-for-user:

- **(a) Ship nothing CUDA; require CUDA Toolkit runtime install.** Smallest download, most
  user friction, version-coupling risk.
- **(b) Ship `ggml-cuda.dll`; require the user to have CUDA runtime** (toolkit or the
  NVIDIA "CUDA runtime" redistributable). Medium.
- **(c) Ship the whole CUDA pack (~500 MB)** including cudart/cublas/cublasLt. Zero user
  friction, huge download. (This is what upstream's 640 MB cuBLAS zip does.)
- **(d) Skip CUDA entirely; Vulkan is the only GPU path.** Smallest GPU pack, universal, a
  little slower. **Recommended default GPU path**; offer CUDA pack only for NVIDIA users who
  want the last ~10%.

---

## 4. What changes in our code (small)

The orchestration code barely moves — that's the point.

- `CMakeLists.txt`: delete the whisper `add_subdirectory` + all `GGML_*`/`CMAKE_CUDA_*`
  settings and the `/Ob1` ICE workaround; add a `third_party/whisper/{include,lib,bin}`
  block mirroring FFmpeg; extend `stage_runtime_dlls` to copy whisper/ggml backend DLLs and
  (optionally) a chosen CUDA pack. Remove the CUDA-copy glob (moves into the GPU pack).
- `transcribe.cpp`: add `ggml_backend_load_all()` + device query/report in `Session::init`;
  keep `whisper_context_params.{use_gpu,gpu_device}` but let ggml's registry pick the device.
- `build.bat`: drop CUDA/nvcc discovery; it becomes a plain MSVC + CMake wrapper. Add a
  `scripts/fetch-deps.ps1` that pulls the prebuilt whisper zip(s) into `third_party/whisper`.
- `patches/whisper.patch` + `third_party/whisper.cpp` submodule: **removed from the default
  build** (kept only if we retain an optional from-source dev build for multi-pass timing).
- No changes to `segment_mode.*`, `audio.*`, `separate.*`, `srt.*`, `timeline.*` logic.

---

## 5. A one-time controlled build of whisper (why it may still be needed)

Consuming stock prebuilt zips is ideal, **but** two things may force a single controlled
build of whisper→DLLs (done once per whisper version, on a dev machine or CI — *never* on
the end user's machine, and *without* CUDA in the main project build):

1. **`GGML_BACKEND_DL=ON`** — needed for clean runtime CPU↔GPU selection. Stock upstream
   zips may be built with the backend statically registered (one zip per backend), not as
   drop-in DLLs. If so, we build once with `-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
   -DBUILD_SHARED_LIBS=ON` and per-backend targets to get `ggml-cpu-*.dll` + `ggml-cuda.dll`
   / `ggml-vulkan.dll`.
2. **Vulkan backend** — not attached to every release; a one-time `-DGGML_VULKAN=ON` build
   (needs only the Vulkan SDK, not CUDA) produces `ggml-vulkan.dll`.

Either way, the *project's* build is CUDA-free and toolkit-free; whisper DLLs are inputs.

---

## 6. Licensing / distribution notes

- **whisper.cpp / ggml:** MIT — bundle freely.
- **FFmpeg (LGPL-2.1+ audio path):** already how we ship; dynamic DLLs satisfy LGPL. Keep
  the relink notice.
- **CUDA runtime (cudart/cublas/cublasLt):** redistributable under the NVIDIA CUDA EULA —
  legal to bundle (that's what upstream's cuBLAS zip does), but this is the size cost.
- **ONNX Runtime (MIT) + DirectML (MS redistributable):** already bundled.
- **Models (Whisper GGUF, MDX-Net UVR):** downloaded on first run today; keep that.

---

## 7. Open questions / tradeoffs to decide

1. **GPU strategy:** Vulkan-first (small, universal, ~10% slower) vs CUDA-first (biggest
   download, fastest on NVIDIA) vs both-as-optional-packs? *Recommendation: base = CPU +
   DirectML isolation; default GPU pack = Vulkan; CUDA as an optional NVIDIA power pack.*
2. **Multi-pass timing patch:** drop it (accept stock timing on the legacy path) or keep an
   optional from-source build for it? If we're committing to segment mode, dropping it is
   cleanest. (Ties to whether multi-pass is retired — see the earlier simplification report.)
3. **Do we need our own controlled whisper build at all,** or can stock upstream zips be
   made to do CPU↔GPU fallback as-is? Needs a spike: drop `ggml-cuda.dll` from the cuBLAS
   zip next to the CPU zip's `whisper.dll` and see if `ggml_backend_load_all()` picks it up.
4. **CUDA version:** upstream prebuilt tops out at CUDA 12.4; we currently build 13.0. 12.4
   runtime + a recent driver is fine, but confirm on the target 3080 Ti driver.
5. **Isolation on CPU-only machines:** DirectML needs a DX12 GPU. On a truly GPU-less box,
   do we fall back to an ONNX **CPU** EP for isolation (slow), or auto-disable isolation and
   transcribe the raw audio? Needs a decision.
6. **First-run downloader vs. full offline installer:** ship one fat installer, or a small
   base that fetches the GPU pack + models on demand?

---

## 8. Phased plan

1. **Spike (½–1 day):** drop stock `whisper-bin-x64` (CPU) + `whisper-cublas-12.4.0`
   `ggml-cuda.dll` side by side; test whether `ggml_backend_load_all()` gives CPU↔CUDA
   fallback without a custom build. Decides Q3/Q7.
2. **Consume prebuilt whisper (CPU) as DLLs:** add `third_party/whisper` (import lib +
   headers + DLLs), rip `add_subdirectory` and all CUDA/ggml build settings out of
   `CMakeLists.txt`, wire `ggml_backend_load_all()` + device reporting. Ship a **CPU-only**
   build end-to-end (validates the whole thin-exe model). Exe shrinks to ~2 MB.
3. **GPU packs:** produce `ggml-vulkan.dll` (and/or the CUDA pack) — from stock zips if the
   spike allows, else one controlled `GGML_BACKEND_DL` build. Verify runtime auto-selection
   on the 3080 Ti and a CPU-only VM.
4. **Productise:** first-run backend/driver detection + optional pack/model downloader;
   tiered zips; README/ARCHITECTURE updates; retire `patches/whisper.patch` +
   `build.bat` CUDA discovery (or gate them behind an optional dev-from-source flag).
5. **Cleanup:** if multi-pass is being retired anyway, remove `timeline.*` (~1,065 lines,
   see prior report) so the shipped surface is just orchestration + segment mode.

---

## 9. Bottom line

- FFmpeg and ONNX/DirectML are **already** runtime DLLs and vendor-neutral; the whole CUDA
  problem is isolated to **whisper transcription**.
- Because **segment mode (default) doesn't use whisper's internal VAD**, we can consume
  **stock prebuilt whisper DLLs** and drop the from-source CUDA build and our patch.
- `GGML_BACKEND_DL` gives **CUDA-optional + automatic CPU fallback** with essentially no
  orchestration code change.
- **Vulkan** collapses the ~530 MB CUDA footprint to a few MB at ~10% speed cost, giving a
  small, universal GPU pack.
- Target end state: a **~90 MB CPU-runnable base** that anyone can unzip and use, plus an
  **additive GPU pack** (Vulkan default; CUDA optional for NVIDIA), and a **CUDA/nvcc-free
  project build**.
