# SRTCreator

A Windows-native, single-purpose command-line utility that generates an `.srt`
subtitle file from **any** movie file (any container, any audio/video codec) as
fast as the hardware allows. Point it at a file, get a synchronized SRT next to
it.

    srt "D:\Movies\Some.Movie.2021.mkv"
    # -> D:\Movies\Some.Movie.2021.srt

---

## 1. Goal & constraints

- **Fastest and lightest** possible. Single native `srt.exe` (C++), no Python,
  no interpreter, no framework runtime.
- **Any input format.** Whatever container/codec the user throws at it must
  decode. This is a hard requirement and it dictates the FFmpeg dependency.
- **GPU-accelerated on NVIDIA.** Target hardware includes an RTX 3080 Ti.
  Maximum speed is the priority; shipping NVIDIA CUDA redistributable DLLs
  alongside the exe is acceptable (see §6, Distribution).
- **Minimal external process dependencies at runtime.** FFmpeg is *statically
  linked in* (audio path only) rather than shelled out to.

Non-goals (for now): GUI, real-time/live captioning, speaker diarization,
translation UI polish (basic translate-to-English is supported by the engine and
exposed as a flag). Burn-in / hardsub is out of scope — we emit `.srt` only.

---

## 2. Architecture (decided)

Three vendored components compiled into **one** executable:

| Component | Role | Integration |
|---|---|---|
| **whisper.cpp** (+ ggml) | ASR: audio -> timestamped text | Vendored source, static link, **CUDA** backend |
| **FFmpeg** (minimized) | Decode any container/codec -> 16 kHz mono f32 PCM | Minimized **static** build, audio path only |
| **App core** (`main.cpp` + glue) | CLI, pipeline, SRT writer, model mgmt | Our code |

Runtime pipeline:

```
input file
  -> libavformat  demux, pick best audio stream
  -> libavcodec   decode audio packets
  -> libswresample resample -> 16 kHz, mono, float32
  -> (optional) Silero VAD: drop non-speech regions
  -> whisper.cpp (CUDA, flash-attn) -> segments with timestamps
  -> SRT writer -> <input>.srt
```

### Why this stack

- **whisper.cpp** is the only high-quality ASR engine genuinely designed to be
  *embedded* in a native C/C++ exe. MIT-licensed, pure C/C++, static-links
  cleanly, has a mature CUDA backend, supports `large-v3-turbo`, flash
  attention, quantized models, word-level timestamps, and Silero VAD.
- **FFmpeg** is the only realistic way to decode *any* audio format
  (AC3/EAC3/DTS/TrueHD/Opus/AAC/FLAC/...) out of *any* container. Single-header
  decoders (minimp3, dr_libs) cannot cover this breadth and are rejected.

### Alternatives considered and rejected

- **insanely-fast-whisper** — NOT a native engine. It is a Python CLI over
  Hugging Face Transformers + PyTorch. Its speed comes from *batched chunked
  inference* + Flash-Attention-2, not from native code. It drags a multi-GB
  Python/PyTorch/CUDA stack, which is the opposite of a lightweight single exe.
  Rejected as a dependency; its flash-attention technique is already available in
  ggml and enabled by default (see §3).
- **faster-whisper / CTranslate2** — CTranslate2 is a fast C++ backend, but it is
  Python-fronted and far less embed-friendly for a single-exe build than
  whisper.cpp. Rejected for embeddability.
- **Cherry-picking FFmpeg source files** — infeasible; FFmpeg's demuxers/decoders
  share deep infrastructure. Correct approach is build-time minimization (§4).
- **Vulkan / DirectCompute backends** — would yield a truly dependency-free exe
  (no redist DLLs), but ~5–15% slower than CUDA. Rejected because the user chose
  maximum speed and accepts shipping CUDA DLLs.

---

## 3. Performance strategy

The big speed levers, in order of impact for typical movie files:

1. **`large-v3-turbo` (q8_0) default model.** 4 decoder layers instead of 32 —
   cuts the dominant serial decode cost while keeping near-`large-v3` accuracy.
   Quantized to `q8_0` for a lighter download and a bit more speed at negligible
   accuracy cost.
2. **Flash attention** — `wparams.flash_attn = true` (ggml's fused
   `ggml_flash_attn_ext` CUDA kernel). ON by default. This is the FA-2 equivalent
   and requires no new code.
3. **Silero VAD** (whisper.cpp built-in) — skip music/silence/action scenes with
   no speech. Often the single biggest real-world win on movies, with no accuracy
   cost. ON by default; tunable.
4. **CUDA graphs** (ggml-cuda) — reduce per-kernel launch overhead. Enabled via
   the CUDA backend.
5. **Quantized default** (`q8_0`) — the shipped `large-v3-turbo` is `q8_0`: less
   VRAM, slightly faster, negligible accuracy loss. Other quant/precision variants
   are downloadable if wanted (§5).

> **Batching is deliberately out of scope.** Chunked-batch decode (the
> insanely-fast trick) adds real complexity — batched KV-cache/EOT orchestration
> in the decode loop — and sacrifices cross-chunk context (accuracy dips, seam
> artifacts). With `large-v3-turbo` + flash-attn + VAD already saturating the win
> for single-file movie transcription, it is not worth the cost. Do not add it.

Rule of thumb target: on a 3080 Ti, `large-v3-turbo` (q8_0) + FA + VAD should
transcribe a 2-hour movie well under realtime.

---

## 4. Build

The top-level `CMakeLists.txt` `add_subdirectory()`s vendored whisper.cpp with
`GGML_CUDA=ON` and links the `whisper` target (which pulls in ggml / ggml-cuda /
cudart / cublas transitively) plus the FFmpeg audio libs into `srt.exe`.
`build.bat` is a thin wrapper: it discovers MSVC (vswhere → VsDevCmd, Axiom-style),
puts CUDA and Ninja on PATH, and runs CMake.

```
build.bat            # configure + build (Release) -> build\srt.exe
build.bat clean      # wipe build\ and rebuild
```

Toolchain on this machine (verified):

- **VS 2022 Build Tools** (MSVC x64). VS 2019 is also installed; `vswhere -latest`
  correctly selects 2022, which CUDA 13 requires.
- **CMake 4.1** and **Ninja** (bundled under
  `<VS>\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja`).
- **CUDA Toolkit 13.0** (`CUDA_PATH` → `...\CUDA\v13.0`).

### 4.1 Generator: use Ninja, not the Visual Studio generator

The VS generator needs CUDA's MSBuild integration (`.props/.targets`) copied into
the VS install; here it is **not** installed (CUDA was installed before/without VS
integration), so the VS generator fails with **"No CUDA toolset found."** Ninja
invokes `nvcc` directly and sidesteps this entirely. `build.bat` forces `-G Ninja`
and prepends the bundled Ninja to PATH. **Do not switch back to the VS generator**
unless the CUDA MSBuild integration is installed.

> Automation note: driving the build by invoking `cmd` non-interactively mangles
> output because **Clink** (a cmd AutoRun add-on) is installed. For scripted
> builds, drive CMake from PowerShell via `Enter-VsDevShell` (see
> `scripts` / dev notes), which Clink does not hook. `build.bat` run normally by
> the user is unaffected.

### 4.2 whisper.cpp (vendored, CUDA)

Git submodule at `third_party/whisper.cpp` (pinned; `git submodule update --init
--recursive`). Built with `GGML_CUDA=ON`, `BUILD_SHARED_LIBS=OFF`,
`CMAKE_CUDA_ARCHITECTURES=86` (Ampere / 3080 Ti). Flash-attn kernels are compiled
by default. API in use (commit 52a939a2): `whisper_context_params.{use_gpu,
flash_attn,gpu_device}`, `whisper_full_params.{language,detect_language,translate,
token_timestamps,vad,vad_model_path,vad_params}`.

### 4.3 FFmpeg audio libraries

**Bootstrap (current):** prebuilt **BtbN LGPL shared** dev libs staged at
`third_party/ffmpeg/{include,lib,bin}` (git-ignored, provided locally). CMake
links `avformat/avcodec/avutil/swresample` import libs and copies the DLLs next to
the exe. This is LGPL (audio-only, no GPL components) and matches our licensing
goal.

**Optimization (later):** `scripts/build-ffmpeg.sh` builds a minimized, static,
audio-only FFmpeg via MSYS2 (`--disable-everything` + explicit audio
demuxers/decoders + `aresample`). Point `FFMPEG_DIR`/`third_party/ffmpeg` at its
output to fold the audio path into the exe. **MSYS2 is not yet installed** — this
is a deferred step, not required for first light.

> Licensing: FFmpeg audio decode/demux is LGPL-2.1+. Static-linking LGPL requires
> providing the means to relink if the exe is distributed. Moot for personal use;
> document if we ever ship.

---

## 5. CLI design

```
srt <input> [options]

Positional:
  <input>                 Path to the movie/audio file.

Common options:
  -o, --output <path>     Output SRT path (default: <input>.srt)
  -m, --model <name|path> Model: tiny|base|small|medium|large-v3|large-v3-turbo
                          (optionally with a quant suffix, e.g.
                          large-v3-turbo-q8_0, large-v3-q5_1) or a path to a
                          .gguf/.bin. Default: large-v3-turbo-q8_0
  -l, --language <code>   Source language (e.g. en, es). Default: auto
      --translate         Translate to English instead of transcribing
      --audio-stream <n>  Pick a specific audio stream index (default: best)
      --duration <sec>    Only transcribe the first <sec> seconds (testing/preview)

Speed / quality:
      --flash-attn        Flash attention (default: ON; --no-flash-attn to off)
      --vad               Silero VAD to skip non-speech (default: ON)
      --threads <N>       CPU threads for pre/post (default: auto)
      --word-timestamps   Emit word-level timing (tighter sync, slower)
      --max-line-length   Wrap subtitles to N chars/line (default 42; 0=off)

Model management:
      --models-dir <path> Where models are stored/downloaded
      --download <name>   Fetch a model and exit

Misc:
      --quiet / --verbose
      --version
```

Behavior notes:

- If no model is present, prompt (or `--download`) to fetch it into
  `--models-dir` (default: next to the exe, or `%LOCALAPPDATA%\SRTCreator\models`).
- Auto-detect language by default; allow override for speed/accuracy.
- Progress to stderr; SRT to file. Exit non-zero on failure with a clear message.

---

## 6. Distribution

Shipped as a folder (or zip) containing:

```
srt.exe
cudart64_*.dll
cublas64_*.dll
cublasLt64_*.dll        (large)
models\                 (optional; or downloaded on first run)
README / LICENSE(S)
```

Runtime requirements for the end user:

- **NVIDIA GPU driver** (provides the CUDA *driver* API). Recent driver
  recommended for CUDA 12.x.
- The **CUDA runtime DLLs are shipped with the exe** (above) — the user does NOT
  need to install the full CUDA Toolkit. Document a fallback: if the app reports a
  missing/incompatible CUDA runtime, either update the GPU driver or install the
  matching **CUDA Toolkit runtime**. Provide the exact CUDA version we built
  against in the README.
- CPU-only fallback is a possible future build flag but not the default.

---

## 7. Repository layout (planned)

```
SRTCreator/
  CLAUDE.md              (this file)
  README.md
  CMakeLists.txt         top-level: whisper.cpp (CUDA) + ffmpeg -> srt.exe
  build.bat              MSVC/Ninja/CUDA discovery + cmake wrapper
  .gitignore
  src/                   (audio+transcribe+srt+models compile into the srtcore lib)
    main.cpp             CLI + orchestration (-> srt.exe)
    audio.{h,cpp}        FFmpeg demux/decode/resample -> f32 16k mono
    transcribe.{h,cpp}   whisper.cpp wrapper, VAD, options, live callbacks
    srt.{h,cpp}          SRT formatting / writing + line wrapping
    models.{h,cpp}       model resolution + download
    gui/main_gui.cpp     Win32 drag-and-drop GUI (-> srtgui.exe)
  third_party/
    whisper.cpp/         vendored submodule (committed)
    ffmpeg/              dev/shared libs: include\ lib\ bin\ (git-ignored, local)
  build/                 cmake/ninja output incl. srt.exe + staged DLLs (ignored)
  models/                downloaded models (gitignored; real store is %LOCALAPPDATA%)
  scripts/
    build-ffmpeg.sh      minimized static build (§4.3, later)
    fetch-models.ps1
```

---

## 8. Implementation milestones

**Status (verified 2026-09-07):** Milestones 1–6 are DONE and working end-to-end.
`srt.exe` builds (Ninja + CUDA 13), and transcribed a real UHD file
(`Transformers.One` — E-AC3 7.1 → mono → VAD → whisper) at **~70× realtime**
(10 min of audio in 8.6 s) with accurate text and timestamps. Remaining: milestone
7 polish + the minimized-static FFmpeg optimization (§4.3).

> **Gotcha (do not regress):** for auto language, set `whisper_full_params.language
> = "auto"` and leave `detect_language = false`. `detect_language = true` is a
> *detect-only* mode that returns zero segments — it silently produces empty SRTs.
> See `src/transcribe.cpp`.

1. **Skeleton + CLI** — arg parsing, resolve input/output paths.
2. **FFmpeg audio path** — `build-ffmpeg.sh`, then `audio.cpp`: open any file,
   select best audio stream, decode, resample to 16 kHz mono f32. Validate on
   mkv/mp4/avi/ts with AAC/AC3/EAC3/DTS/Opus/FLAC.
3. **whisper.cpp integration** — link CUDA build, run `whisper_full` on the PCM,
   dump raw segments. Confirm GPU is used.
4. **SRT writer** — correct indexing, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, UTF-8,
   sensible line wrapping/segment splitting.
5. **Speed defaults** — enable flash-attn, VAD, CUDA graphs; default
   `large-v3-turbo-q8_0`. Benchmark realtime factor on a full movie.
6. **Model management** — auto-download, `--models-dir`, quant/precision variants.
7. **Polish** — language/translate flags, word-timestamps, progress UI, robust
   error handling, packaging (§6).

---

## 9. Environment notes (this machine)

- `ffmpeg.exe` is installed at `L:\ffmpeg\bin\ffmpeg.exe`. We do **not** depend on
  it at runtime (we static-link the audio libs), but it is handy for probing test
  files (`ffprobe`) and as a sanity oracle during development.
- GPU: **NVIDIA GeForce RTX 3080 Ti** (12 GB, Ampere, **SM 8.6**) — CUDA target.
  Build ggml-cuda for `CMAKE_CUDA_ARCHITECTURES=86`. Ample VRAM for any model;
  default ships `large-v3-turbo-q8_0`.
- Platform: Windows 11. Shell: PowerShell (primary); Bash/MSYS2 for the FFmpeg
  build.
- Not currently a git repo — `git init` when ready to start committing.

---

## 10. Working agreements for Claude

- Prefer editing the vendored build config over patching vendored source; keep
  `third_party/` as close to upstream as possible so it can be updated.
- Keep the audio pipeline codec-agnostic — never special-case a container/codec
  when FFmpeg already handles it.
- Any change to model defaults, flags, or CUDA version must be reflected in this
  file and the README.
- Benchmark speed claims with an actual realtime factor on a real movie before
  asserting them.

## 11. GUI (`srtgui.exe`, `gui` branch)

Minimal Win32 drag-and-drop front end over the shared `srtcore` library. Drop a
file -> worker thread runs `audio::decode` + `transcribe::run` + `srt::write`;
subtitles stream into a read-only EDIT panel with a progress bar. All CLI toggles
are checkboxes (Translate/VAD/Flash/Word-timestamps/Wrap) plus Model + Language
dropdowns.

- Live updates use `transcribe::Options.on_segment` / `on_progress`, which wrap
  whisper's `new_segment_callback` / `progress_callback`. These fire on the worker
  thread and are marshaled to the UI thread via `PostMessage` (`WM_APP_*`).
- `on_segment` is display-only; `transcribe::run` still fills the segment vector
  used for the file (don't push in both or lines double).
- No separate manifest: comctl6 + visual styles via a `#pragma comment(linker,...)`
  manifestdependency; `SetProcessDPIAware()` at startup.
- Known rough edge: model download uses `curl` via `std::system`, which flashes a
  console window from the GUI. Fix later with `CreateProcess` + `CREATE_NO_WINDOW`
  (or an in-process HTTP download).
