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

Toolchain required:

- **Visual Studio Build Tools** (MSVC) + **CMake** (>= 3.21)
- **CUDA Toolkit** (>= 12.x) — provides `nvcc`, cuBLAS; needed to compile the
  ggml-cuda backend
- **MSYS2** — required to run FFmpeg's `configure` even when targeting MSVC
- Vendored sources: `third_party/whisper.cpp`, `third_party/ffmpeg`

### 4.1 Minimized static FFmpeg (audio-only)

Build once, from an MSYS2 shell with the MSVC environment loaded. The point is to
compile only the audio demuxers/decoders + resampler as **static** libs. Adjust
the enabled demuxers/decoders to taste, but keep the common movie audio codecs.

```sh
./configure \
  --toolchain=msvc \
  --disable-everything --disable-programs --disable-doc \
  --enable-static --disable-shared \
  --enable-protocol=file \
  --enable-demuxer=matroska,mov,mpegts,avi,flv,ogg,wav,mp3,flac,aac,ac3,dts \
  --enable-decoder=aac,aac_latm,ac3,eac3,dca,truehd,mlp,mp3,mp2,opus,vorbis,flac,pcm_s16le,pcm_s24le,pcm_f32le,alac,wmav1,wmav2 \
  --enable-parser=aac,ac3,dca,flac,mpegaudio,opus,vorbis \
  --enable-filter=aresample \
  --disable-encoders --disable-muxers --disable-bsfs \
  --disable-network --disable-avdevice --disable-postproc --disable-swscale \
  --prefix=../ffmpeg-min
make -j && make install
```

Produces static `libavformat`, `libavcodec`, `libavutil`, `libswresample` under
`ffmpeg-min/`. These are linked into `srt.exe`.

> Licensing: FFmpeg core is LGPL-2.1+. Keeping to audio decode/demux avoids GPL
> components. Static-linking LGPL code requires providing the means to relink
> (object files) if the exe is distributed. Document this if we ever ship it.

### 4.2 whisper.cpp with CUDA

Built as part of our CMake project (or prebuilt static libs), with:

```
-DGGML_CUDA=ON
```

Ensure flash-attn kernels are compiled (default in recent ggml-cuda).

### 4.3 App CMake

One `CMakeLists.txt` that:

- adds whisper.cpp + ggml (CUDA) as static targets
- imports the minimized FFmpeg static libs from `ffmpeg-min/`
- builds `src/main.cpp` + glue into `srt.exe`
- links: whisper, ggml, ggml-cuda, cudart, cublas, cublasLt, avformat,
  avcodec, avutil, swresample, plus Windows/CRT libs

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

Speed / quality:
      --flash-attn        Flash attention (default: ON; --no-flash-attn to off)
      --vad               Silero VAD to skip non-speech (default: ON)
      --threads <N>       CPU threads for pre/post (default: auto)
      --word-timestamps   Emit word-level timing (tighter sync, slower)

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
  CMakeLists.txt
  src/
    main.cpp             CLI + orchestration
    audio.{h,cpp}        FFmpeg demux/decode/resample -> f32 16k mono
    transcribe.{h,cpp}   whisper.cpp wrapper, VAD, options
    srt.{h,cpp}          SRT formatting / writing
    models.{h,cpp}       model resolution + download
  third_party/
    whisper.cpp/         vendored
    ffmpeg/              vendored source (for the minimized build)
  ffmpeg-min/            build output: minimized static libs + headers
  models/                downloaded models (gitignored)
  scripts/
    build-ffmpeg.sh      the §4.1 minimized build
    fetch-models.ps1
```

---

## 8. Implementation milestones

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
