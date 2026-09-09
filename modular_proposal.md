# Modular / Productisation Refactor Proposal

**Status:** exploration + plan (branch `module_refactor`). **Decisions locked 2026-09-09 — see §11.** Not yet implemented.

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
3. ~~**Do we need our own controlled whisper build at all?**~~ **RESOLVED — see §10.**
   Stock upstream zips are built with `GGML_BACKEND_DL=ON`; `ggml-cuda.dll` is a drop-in.
   No custom build needed for CPU+CUDA. (Vulkan still needs a one-time build.)
4. **CUDA version:** upstream prebuilt tops out at CUDA 12.4; we currently build 13.0. 12.4
   runtime + a recent driver is fine, but confirm on the target 3080 Ti driver.
5. **Isolation on CPU-only machines:** DirectML needs a DX12 GPU. On a truly GPU-less box,
   do we fall back to an ONNX **CPU** EP for isolation (slow), or auto-disable isolation and
   transcribe the raw audio? Needs a decision.
6. **First-run downloader vs. full offline installer:** ship one fat installer, or a small
   base that fetches the GPU pack + models on demand?

---

## 11. Decisions (locked 2026-09-09) — Path A, split build

- **GPU strategy: keep CUDA (proven, default) + Vulkan (validated fallback) + CPU, all as
  `GGML_BACKEND_DL` plugin DLLs at ONE ggml version, built from the pinned submodule.**
  Runtime auto-selects the best available (CUDA → Vulkan → CPU); the GUI shows a radio button
  per *detected* backend. CUDA moves from static-in-exe to `ggml-cuda.dll` (still prebuilt &
  proven — just a plugin). Building from one submodule commit guarantees ABI coherence, which
  mixing prebuilt zips from different tags does NOT (confirmed: whisper-b4938 vs llama-b10872
  ship different-sized `ggml-base.dll` → won't safely co-load).
- **Remove the multi-pass pipeline AND drop `patches/whisper.patch`.** Delete
  `timeline.{h,cpp}` (~1,065 lines) and the legacy Pass 2/3 code paths, flags, and UI.
  Segment mode is the sole pipeline; it needs neither the patch nor whisper's internal VAD.

### Split into two builds (the structural key)
Do NOT keep `add_subdirectory(whisper)` in the app build. Two independent builds:

1. **Deps build (rare — only on a whisper version bump):** `scripts/build-whisper-dlls.ps1`
   compiles the vendored submodule with
   `-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON
   -DGGML_VULKAN=ON` and stages the coherent DLL set + import lib + headers into
   `third_party/whisper/{bin,lib,include}`. **Needs CUDA Toolkit + Vulkan SDK — build machine
   / CI only.**
2. **App build (common):** MSVC-only. `CMakeLists.txt` consumes `third_party/whisper` exactly
   like FFmpeg (import lib + headers), no `add_subdirectory`, **no CUDA/nvcc/Vulkan needed**.
   Exe drops from 67 MB to ~2 MB. Contributors hacking on orchestration need only MSVC.

Result: reproducible from git (one pinned submodule commit → the whole coherent DLL set),
proven CUDA retained, Vulkan + CPU added, all runtime-selectable, thin app build.

### Build-machine requirements (users need NONE of these)
- **CUDA Toolkit** (have it — 13.0) for `ggml-cuda.dll`.
- **Vulkan SDK** (one-time LunarG install; `glslc`/`vulkan-shaders-gen`) for `ggml-vulkan.dll`.
  End users need only their GPU driver (`vulkan-1.dll` ships with it).

### Execution order
1. **Phase 1 — Remove multi-pass (buildable now, no new deps).** Delete `timeline.{h,cpp}`;
   strip the Pass 2/3 blocks, `--multipass`/`--legacy`/`--no-infill`/`--pause-split`/
   `--timeline-json` flags, the GUI "Multi-pass mode" checkbox, and the timeline-summary code.
   Segment mode becomes the only path. Still builds whisper from source (unchanged CMake).
2. **Phase 2a — Vulkan smoke test (MILESTONE 1, gates the whole GPU plan).** Build the
   coherent DLL set once (`build-whisper-dlls.ps1`), run one Tron transcription on **Vulkan**,
   confirm output matches CUDA/CPU and it's genuinely GPU-fast (no silent per-op CPU
   fallback). Commit to Vulkan on evidence, not assumption.
3. **Phase 2b — Split-build swap.** Add the `third_party/whisper` consumption block; remove
   `add_subdirectory` + all `GGML_*`/CUDA settings from `CMakeLists.txt`; wire
   `ggml_backend_load_all()` + generic device enumeration/reporting in `transcribe.cpp`; make
   the GUI backend selector **data-driven** (radio per detected device, not hardcoded);
   retire the submodule-in-app-build + patch + `build.bat` CUDA discovery.
4. **Phase 3 — Productise + CI (see §12).** First-run backend/driver detection, tiered zips,
   GitHub Actions release, licenses, README/ARCHITECTURE rewrite.

---

## 12. Official releasable bundle via GitHub Actions

The split-build maps almost 1:1 onto CI, but going "officially releasable" forces a few
concrete changes to how the repo is set up today.

### 12.1 What CI needs that we don't have yet
- **All deps must be *fetchable by pinned version* — today they're "provided locally,
  git-ignored".** FFmpeg (`third_party/ffmpeg`), ONNX Runtime + DirectML
  (`third_party/onnxruntime`) and the built whisper DLLs are local. A clean CI checkout has
  none of them. So we must add **`scripts/fetch-deps.ps1`** that downloads *pinned* FFmpeg
  (BtbN LGPL shared) and ONNX-Runtime-DirectML by exact version/URL, plus a pinned models
  step if we ever bundle them. This is the single biggest prerequisite change.
- **A version/manifest file** recording pinned versions of every dependency + the whisper
  submodule SHA + each backend DLL, surfaced in `--version` / GUI About and the release notes.

### 12.2 The workflow shape (mirrors whisper.cpp's own `release.yml`)
- **Job `whisper-dlls` (`windows-2022`, heavy, cached):** checkout with submodules; install
  **CUDA Toolkit** (via NVIDIA redist archives, exactly as whisper.cpp's release CI does) and
  the **Vulkan SDK**; run `build-whisper-dlls.ps1`; upload the DLL set as an artifact. **Cache
  keyed by the whisper submodule SHA + flags**, so it only rebuilds on a version bump (CUDA
  kernel compile is 20–40 min).
- **Job `app` (`windows-2022`, fast, MSVC-only):** `fetch-deps.ps1`; download the
  `whisper-dlls` artifact; build `srt.exe` / `srtgui.exe`; assemble bundles.
- **Job `release` (on tag push):** zip the tiered packages, create a GitHub Release, upload
  assets, attach `LICENSES/` and the manifest.

### 12.3 The hard limitation: GitHub-hosted runners have NO GPU
CI can **compile and package** CUDA/Vulkan DLLs, but it **cannot functionally test them on a
GPU**. So:
- CPU-path smoke tests (decode → segment transcribe → SRT) run on hosted runners.
- **GPU validation (the Phase-2a Vulkan smoke test, CUDA correctness) needs a self-hosted
  runner with a GPU** — e.g. register the 3080 Ti machine as a self-hosted runner — or a
  documented manual gate before publishing a release. This is a real constraint to plan for,
  not a blocker.

### 12.4 Licensing for public distribution (stricter than personal use)
- Ship a **`LICENSES/`** dir: MIT (whisper/ggml, ONNX Runtime), **LGPL-2.1+ FFmpeg** with the
  relink offer/notice (dynamic DLLs satisfy it, but the notice is required), NVIDIA CUDA EULA
  (if a CUDA pack is shipped), DirectML redistributable terms.
- Keep **models downloaded on first run** (not bundled) to avoid shipping model weights.

### 12.5 Other "official" polish (optional, flag for decision)
- **Code signing (Authenticode):** unsigned exes trigger SmartScreen. A signing cert (cost +
  a CI secret) removes the warning. Optional but expected of a real release.
- **Version resource** embedded in the exes; semantic-version tags drive releases.
- **Tiered packaging** decision (§7 Q6): one fat installer vs a small base + on-demand GPU
  pack / model downloader.

### 12.6 How CI changes what we build (net)
It mostly *validates* the split-build, but it adds these must-dos to the plan:
1. `scripts/fetch-deps.ps1` (pinned FFmpeg + ONNX/DirectML) — required for any clean build.
2. `scripts/build-whisper-dlls.ps1` must be **fully scriptable / parameterized** (flags,
   output dir) and self-contained (submodule + toolkit installs) so a runner can drive it.
3. A **dependency manifest** (pinned versions) committed to the repo.
4. A **GPU test gate** via self-hosted runner or manual sign-off before publishing.
5. `LICENSES/` + release notes generation.

None of these change the *architecture* — they harden it into something reproducible on a
clean machine, which is exactly what "officially releasable" means.

---

## 8. Phased plan (original, superseded by §11 for the chosen path)

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

## 10. Spike results (verified 2026-09-09)

Investigated the "do we need our own whisper build" question against the **authoritative
source — whisper.cpp's release CI** (`.github/workflows/release.yml`) plus our pinned ggml
headers. Findings:

- **Both the CPU and cuBLAS Windows-x64 release jobs use identical ggml flags:**
  `-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF`
  (the cuBLAS job adds `-DGGML_CUDA=ON`). So every shipped x64 zip already uses dynamic
  backend loading and CPU-variant selection.
- **`ggml-cuda.dll` is therefore a drop-in backend.** The cuBLAS job builds the same
  `whisper.dll` / `ggml.dll` / `ggml-base.dll` / `ggml-cpu*.dll` as the CPU zip, plus
  `ggml-cuda.dll`, and its "Copy CUDA DLLs" step bundles the entire CUDA runtime
  (`$CUDA_PATH\bin\*.dll` → cudart, cublas, cublasLt, nvrtc…) — which is why that zip is
  640 MB (12.4) / 257 MB (11.8). **End users need only an NVIDIA driver, not the Toolkit.**
- **Conclusion: no custom whisper build is required for CPU+CUDA.** Take the 8 MB CPU zip as
  the base; add `ggml-cuda.dll` + the CUDA runtime DLLs from the cuBLAS zip as the NVIDIA
  GPU pack. `ggml_backend_load_all()` picks CUDA when the DLL + driver are present, else CPU.
- **Our pinned ggml (`v1.9.3-182`) exposes the needed API:** `ggml_backend_load_all()`,
  `ggml_backend_load(path)`, `ggml_backend_load_all_from_path(dir)`, and device enumeration
  (`ggml_backend_dev_count/get/name`, `ggml_backend_dev_get_props`) for reporting the active
  backend. So the runtime detection/fallback/report code is straightforward.
- **Vulkan:** there is **no** stock Windows Vulkan release artifact (no `windows-vulkan` job
  in the CI). A Vulkan pack requires a one-time `-DGGML_VULKAN=ON` build (Vulkan SDK only,
  no CUDA) — still a CUDA-free, toolkit-free path for the *project*.
- **Version hygiene:** upstream prebuilt CUDA tops out at **CUDA 12.4** (we build 13.0
  today). 12.4 runtime + a current driver is fine on the 3080 Ti; match the whisper DLL
  version to the headers we compile against (pin a specific release, mirror it locally like
  FFmpeg).

**Residual risk (small):** I could not run the combined DLLs here — the sandbox blocks
GitHub release-asset downloads (they redirect to `objects.githubusercontent.com`), and GPU
selection can't be validated without the hardware. The CI flags make CPU↔CUDA drop-in
near-certain, but the final confirmation is a 10-minute physical test: unzip the CPU + a
`ggml-cuda.dll` next to a tiny test exe on the 3080 Ti and confirm `ggml_backend_load_all()`
selects CUDA.

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
