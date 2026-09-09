# SRTCreator

A fast, Windows-native command-line tool that generates an `.srt` subtitle file
from **any** movie or audio file. It decodes with FFmpeg and transcribes with
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), handling any
container/codec you throw at it and running far faster than real time.

```
srt "D:\Movies\Some.Movie.mkv"
# -> D:\Movies\Some.Movie.srt
```

The transcription engine ships as runtime backend plugins, so the app runs on
**CPU anywhere**, and lights up **GPU acceleration** when you drop in a GPU pack —
**NVIDIA (CUDA)** or the vendor-neutral **Vulkan** (any modern GPU). On an RTX
3080 Ti the CUDA path transcribes a 2-hour movie in well under real time.

> How it fits together — the whisper/ggml backend-DL DLLs, ONNX/DirectML isolation,
> and FFmpeg — see [`ARCHITECTURE.md`](ARCHITECTURE.md). Pinned component versions
> are in [`dependencies.json`](dependencies.json).

---

## Features

- **Any input** — MP4, MKV, AVI, TS, MOV, WebM, … with AAC, AC3/E-AC3, DTS,
  TrueHD, Opus, FLAC, MP3, PCM and more (multichannel audio can extract the
  dialogue front-center channel).
- **GPU vocal isolation** — MDX-Net (Kim_Vocal_2) via ONNX Runtime **DirectML**,
  running on any Direct3D-12 GPU, strips musical score and effects before ASR.
- **Whisper ASR with segment-mode VAD** — `large-v3-turbo` with flash attention.
  Standalone Silero VAD partitions the track into speech regions and transcribes
  each in isolation, so cue timing tracks the real onset with no VAD-seam artifacts.
- **DTW word-level timing** — per-token acoustic alignment tightens each cue.
- **Runtime backend selection** — CUDA → Vulkan → CPU, auto-detected at startup
  (`ggml_backend_load_all`). No GPU? It runs on CPU.
- **Automatic language detection & translation** (`--translate`).
- **Readable line wrapping** (Netflix-style 42 chars/line by default).
- **Self-contained** — models download automatically on first use.

---

## Quick start (use the release zips)

1. Unzip **`SRTCreator-vX.Y.Z-win64-base.zip`** anywhere. This runs on any 64-bit
   Windows 10/11 — CPU transcription plus DirectML vocal isolation, no GPU story
   required.
2. *(Optional, for GPU speed)* unzip **one** GPU pack **into the same folder**,
   next to `srt.exe`:
   - **`...-cuda-pack.zip`** — NVIDIA GPUs (needs only an NVIDIA driver; the CUDA
     runtime is included). Fastest.
   - **`...-vulkan-pack.zip`** — any modern NVIDIA/AMD/Intel GPU (needs only your
     GPU driver). Small; a little slower than CUDA.

   The app detects the dropped-in backend on startup and uses it automatically.
3. Run it from a terminal:
   ```
   srt.exe "C:\path\to\movie.mkv"
   ```
   The first run downloads the speech model (~830 MB), the VAD model, and the
   vocal-isolation model into `%LOCALAPPDATA%\SRTCreator\models`. Subsequent runs
   start instantly.

The subtitle file is written next to the input (`movie.srt`) unless you pass `-o`.
On startup the app prints the compute backends it found, e.g.
`[srt] compute backends: CUDA0 (GPU), Vulkan0 (GPU), CPU (CPU) -> GPU`.

---

## GUI (drag-and-drop)

`srtgui.exe` is a clean, minimal window: **drag a media file onto it** and it
decodes, isolates vocals, transcribes, and saves `<input>.srt` next to the file.

- **Checkboxes**: **Translate → English**, **Center channel (dialogue)** (5.1/7.1
  front-center extraction), **Dump audio (debug)** (export the 16 kHz vocal stem),
  **Debug report**.
- **Dropdowns**: **Model** (Whisper), **Language**, **Vocal** (Kim_Vocal_2 default).
- **Live transcript panel** with real-time decoding and per-phase timing (`hh:mm:ss`).

Flash attention, DTW word timestamps and 42-char wrapping are always on.
No console needed — just double-click `srtgui.exe`.

## Usage (CLI)

```
srt <input> [options]
```

`<input>` is the path to any movie/audio file.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <path>` | `<input>.srt` | Where to write the SRT file. |
| `-m, --model <name\|path>` | `large-v3-turbo-q8_0` | Model name (e.g. `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, optionally with a quant suffix like `-q8_0`/`-q5_1`), **or** a path to a `.bin`/`.gguf` file. Named models auto-download. |
| `-l, --language <code>` | `auto` | Source language as an ISO code (`en`, `es`, `fr`, `de`, `ja`, …) or `auto` to detect. |
| `--translate` | off | Translate speech to **English** subtitles (works with `auto`). Requires a multilingual model. |
| `--audio-stream <n>` | best | Pick a specific audio stream index (for files with multiple audio tracks). |
| `--duration <sec>` | whole file | Only transcribe the first `<sec>` seconds. Handy for a quick preview/test. |
| `--max-line-length <n>` | `42` | Wrap subtitle text to at most `n` characters per line. `0` disables wrapping. |
| `--center` | off | Isolate the Front-Center channel (dialogue) on 5.1/7.1 before isolation. Default isolates from the full stereo downmix. |
| `--vocal-model <name>` | Kim_Vocal_2 | Separation model: `Kim_Vocal_2`, `UVR-MDX-NET-Inst_HQ_3`, `UVR_MDXNET_KARA_2`. |
| `--no-flash-attn` | flash on | Disable flash attention (rarely needed). |
| `--no-word-timestamps` | word-ts on | Disable DTW word-level timing alignment. |
| `--time-offset <sec>` | `0` | Shift every cue by a constant (`+` later, `-` earlier). |
| `--dump-audio [path]` | off | Export the exact 16 kHz vocal audio fed to Whisper as a WAV. |
| `--threads <n>` | auto | CPU worker threads for pre/post-processing. |
| `--models-dir <path>` | `%LOCALAPPDATA%\SRTCreator\models` | Where models are stored/downloaded. |
| `--download <name>` | — | Download a model and exit (e.g. `srt --download large-v3`). |
| `--debug` | off | Write a detailed timing diagnostics report (`<output>.debug.txt`). |
| `--verbose` | off | Print verbose progress and Whisper internals. |
| `-h, --help` | — | Show help. |

### Examples

```bat
:: Basic - vocal isolation, Whisper large-v3-turbo, segment-mode VAD
srt "D:\Movies\movie.mkv"

:: Non-English film -> English subtitles
srt "D:\Movies\film.fr.mkv" --translate

:: Quick 2-minute preview
srt movie.mp4 --duration 120 -o preview.srt

:: Extract the 5.1 center channel before isolating
srt movie.mkv --center

:: Nudge subtitles 0.3s later if there is an external delay
srt movie.mkv --time-offset 0.3
```

### Debugging sync & audio

- **`--dump-audio [path]`** writes the precise 16 kHz mono vocal stem that whisper
  transcribes to a `.wav` (next to `srt.exe` by default) so you can listen and
  judge whether separation cleaned up the dialogue.
- **`--time-offset <sec>`** shifts every cue by a fixed amount. To measure a
  consistent lead/lag, compare against a film's embedded English subtitles:
  `ffmpeg -i movie.mkv -map 0:s:m:language:eng -c:s srt ref.srt`.

### Models

Models are whisper.cpp `ggml` files, fetched automatically from Hugging Face on
first use and cached in `--models-dir`.

| Model | Size | Notes |
|-------|------|-------|
| `large-v3-turbo-q8_0` *(default)* | ~830 MB | Near-large-v3 accuracy, much faster. Best general choice. |
| `large-v3` | ~3 GB | Highest accuracy, slower. |
| `medium` | ~1.5 GB | Good balance, lighter. |
| `small` / `base` / `tiny` | 75–500 MB | Fastest, lower accuracy. Add `.en` (e.g. `base.en`) for English-only. |

---

## Requirements

**To run:** any 64-bit Windows 10/11. The base zip runs on CPU with no GPU
requirement. For GPU acceleration, add a GPU pack:

- **CUDA pack** — an NVIDIA GPU with a recent driver (the CUDA runtime DLLs are in
  the pack; the CUDA Toolkit is not needed).
- **Vulkan pack** — any modern GPU with an up-to-date driver (`vulkan-1.dll` ships
  with the driver).

Vocal isolation uses DirectML and runs on any Direct3D-12 GPU; on a GPU-less box it
falls back appropriately.

---

## Building from source

The build is **split** so contributors hacking on the app need only MSVC — no
CUDA, no Vulkan SDK.

### App build (common — MSVC only)

Prerequisites: **Visual Studio 2022 Build Tools** (MSVC v143 C++ x64; Ninja + CMake
come with the "C++ CMake tools" component) and **Git**.

```
git clone <your-repo-url> SRTCreator
cd SRTCreator

:: 1. Fetch the pinned FFmpeg + ONNX/DirectML dev libraries into third_party\
powershell -ExecutionPolicy Bypass -File scripts\fetch-deps.ps1

:: 2. Ensure the prebuilt whisper backend DLLs are present in third_party\whisper\
::    (produced by the deps build below; on a fresh clone, run that once first)

:: 3. Build (MSVC only - no CUDA/nvcc)
build.bat            :: -> build\srt.exe, build\srtgui.exe (+ staged runtime DLLs)
build.bat clean      :: wipe build\ and rebuild
```

The exes are thin (~0.6 MB); all heavy code lives in the staged DLLs
(`whisper.dll`, `ggml*.dll`, the GPU backends, FFmpeg, ONNX/DirectML).

### Deps build (rare — only on a whisper version bump)

Produces the coherent `GGML_BACKEND_DL` DLL set (CPU all-variants + CUDA + Vulkan)
from the pinned `third_party/whisper.cpp` submodule and stages it into
`third_party/whisper/{bin,lib,include}`. **Build machine / CI only** — needs the
**CUDA Toolkit** and the **Vulkan SDK**.

```
git submodule update --init --recursive
powershell -ExecutionPolicy Bypass -File scripts\build-whisper-dlls.ps1
```

(`-Backends cuda` builds without the Vulkan SDK; `-Jobs N` caps parallelism if the
CUDA assembler `ptxas` hits a flaky crash — the script auto-retries regardless.)

---

## Packaging a release

```
powershell -ExecutionPolicy Bypass -File scripts\package.ps1
```

Produces the tiered `dist\` zips from a prior build:

| Zip | Size | Contents |
|-----|------|----------|
| `SRTCreator-v<ver>-win64-base.zip` | ~58 MB | exes + whisper/ggml (CPU) + FFmpeg + ONNX/DirectML + VC++ runtime + licenses |
| `SRTCreator-v<ver>-win64-cuda-pack.zip` | ~420 MB | `ggml-cuda.dll` + CUDA runtime |
| `SRTCreator-v<ver>-win64-vulkan-pack.zip` | ~17 MB | `ggml-vulkan.dll` |

Whisper models are **not** bundled (they download on first run). CI in
`.github/workflows/release.yml` builds and publishes these on a version tag.

---

## Troubleshooting

- **Runs on CPU when you expected GPU** — check the startup `compute backends` line.
  Make sure a GPU pack's DLL sits next to `srt.exe` and your GPU driver is current.
- **Empty SRT / 0 segments** — usually no detectable speech in the range, or an
  English-only (`.en`) model on non-English audio. Try the default model or `-l`.
- **Wrong audio track** — use `--audio-stream <n>` (list with `ffprobe -show_streams`).
- **Model download fails** — check your network, or pre-fetch with `scripts\fetch-models.ps1`.

---

## Licensing

SRTCreator's own code, plus bundled components: whisper.cpp/ggml (MIT), FFmpeg
(LGPL-2.1+, audio-only, unmodified shared libs, with a relink notice), ONNX Runtime
(MIT), DirectML (Microsoft redistributable), NVIDIA CUDA runtime (redistributable,
CUDA pack only), and the Microsoft VC++ runtime. See
[`LICENSES/THIRD_PARTY_NOTICES.md`](LICENSES/THIRD_PARTY_NOTICES.md).
