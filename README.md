# SRTCreator

A fast, Windows-native command-line tool that generates an `.srt` subtitle file
from **any** movie or audio file. It decodes with FFmpeg and transcribes with
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) on the GPU (NVIDIA CUDA),
so it handles any container/codec you throw at it and runs far faster than
real time.

```
srt "D:\Movies\Some.Movie.mkv"
# -> D:\Movies\Some.Movie.srt
```

On an RTX 3080 Ti it transcribes roughly **70× faster than real time**
(a 2-hour movie in ~1.5 minutes) with the default model.

---

## Features

- **Any input** — MP4, MKV, AVI, TS, MOV, WebM, … with AAC, AC3/E-AC3, DTS,
  TrueHD, Opus, FLAC, MP3, PCM and more (multichannel audio is downmixed to mono).
- **GPU-accelerated** whisper `large-v3-turbo` (quantized) with flash attention.
- **Automatic language detection.**
- **Translation** — transcribe a non-English film straight to English subtitles
  (`--translate`).
- **Voice Activity Detection (VAD)** to skip music/silence (on by default).
- **Readable line wrapping** (Netflix-style 42 chars/line by default).
- **Self-contained** — models download automatically on first use.

---

## Quick start (use the release zip)

1. Unzip `SRTCreator-vX.Y.Z-win64-cuda.zip` anywhere.
2. Make sure you have a reasonably recent **NVIDIA GPU driver** installed
   (nothing else — the CUDA runtime DLLs are included).
3. Run it from a terminal:
   ```
   srt.exe "C:\path\to\movie.mkv"
   ```
   The first run downloads the speech model (~830 MB) and the VAD model into
   `%LOCALAPPDATA%\SRTCreator\models`. Subsequent runs are instant to start.

The subtitle file is written next to the input (`movie.srt`) unless you pass
`-o`.

---

## Usage

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
| `--translate` | off | Translate speech to **English** subtitles (works with `auto` — detects the source language, writes English). Requires a multilingual model (the default is fine; `.en` models can't translate). |
| `--audio-stream <n>` | best | Pick a specific audio stream index (for files with multiple audio tracks). Use `ffprobe` to list them. |
| `--duration <sec>` | whole file | Only transcribe the first `<sec>` seconds. Handy for a quick preview/test. |
| `--max-line-length <n>` | `42` | Wrap subtitle text to at most `n` characters per line (balanced across two lines when it fits). `0` disables wrapping. |
| `--no-vad` | VAD on | Disable Voice Activity Detection. VAD skips non-speech and is recommended; disable only to debug. |
| `--no-flash-attn` | flash on | Disable flash attention (rarely needed). |
| `--word-timestamps` | off | Compute word-level timestamps (tighter sync, slightly slower). |
| `--threads <n>` | auto | CPU worker threads for pre/post-processing. |
| `--models-dir <path>` | `%LOCALAPPDATA%\SRTCreator\models` | Where models are stored/downloaded. |
| `--download <name>` | — | Download a model and exit (e.g. `srt --download large-v3`). |
| `--verbose` | off | Print whisper progress and detail. |
| `-h, --help` | — | Show help. |

### Examples

```bat
:: Basic - auto language, default model, VAD + wrapping on
srt "D:\Movies\movie.mkv"

:: Non-English film -> English subtitles
srt "D:\Movies\film.fr.mkv" --translate

:: Force a language (skips detection; a touch faster / more reliable)
srt movie.mp4 -l en

:: Maximum accuracy with the full large-v3 model
srt movie.mp4 -m large-v3

:: Quick 2-minute preview
srt movie.mp4 --duration 120 -o preview.srt

:: Custom output path and wider lines
srt movie.mp4 -o subs\movie.srt --max-line-length 50

:: Pick the second audio track
srt movie.mkv --audio-stream 2

:: Pre-download a model
srt --download large-v3-turbo-q8_0
```

### Models

Models are whisper.cpp `ggml` files, fetched automatically from Hugging Face on
first use and cached in `--models-dir`. Rough sizes / trade-offs:

| Model | Size | Notes |
|-------|------|-------|
| `large-v3-turbo-q8_0` *(default)* | ~830 MB | Near-large-v3 accuracy, much faster. Best general choice. |
| `large-v3` | ~3 GB | Highest accuracy, slower. |
| `medium` | ~1.5 GB | Good balance, lighter. |
| `small` / `base` / `tiny` | 75–500 MB | Fastest, lower accuracy. Add `.en` (e.g. `base.en`) for English-only. |

---

## Requirements

**To run:** an NVIDIA GPU with a recent driver. The CUDA runtime and all other
DLLs are bundled in the release zip — you do **not** need the CUDA Toolkit or the
VC++ Redistributable installed.

> If you built it yourself and see a missing-DLL error, either use the packaged
> zip (`scripts\package.ps1`) or install the NVIDIA CUDA runtime / VC++ 2015-2022
> x64 Redistributable.

---

## Building from source

### Prerequisites

- **Visual Studio 2022 Build Tools** with the *MSVC v143 C++ x64* toolset
  (Ninja and CMake ship with its "C++ CMake tools" component).
- **CUDA Toolkit 13.x** (provides `nvcc`; `CUDA_PATH` should point at it).
- **FFmpeg dev/shared libraries** (see below).
- **Git** (whisper.cpp is a submodule).

### 1. Clone with the submodule

```
git clone <your-repo-url> SRTCreator
cd SRTCreator
git submodule update --init --recursive
```

### 2. Provide FFmpeg dev libraries

The build links FFmpeg's import libs and ships its DLLs. Download an **LGPL
shared** build (e.g. BtbN's `ffmpeg-n*-win64-lgpl-shared-*.zip`) and place its
folders so the layout is exactly:

```
third_party\ffmpeg\
  include\   (libavformat\, libavcodec\, libavutil\, libswresample\, ...)
  lib\       (avformat.lib, avcodec.lib, avutil.lib, swresample.lib, ...)
  bin\       (avformat-*.dll, avcodec-*.dll, avutil-*.dll, swresample-*.dll, ...)
```

`third_party\ffmpeg\` is git-ignored — it's provided locally, not committed.

### 3. Build

```
build.bat            :: configure + build (Release) -> build\srt.exe
build.bat clean      :: wipe build\ and rebuild
```

`build.bat` locates MSVC (via `vswhere` → `VsDevCmd`), puts CUDA and **Ninja** on
PATH, and runs CMake. The exe and all required runtime DLLs (FFmpeg + CUDA) are
copied into `build\`.

> **Why Ninja, not the Visual Studio generator?** The VS generator needs CUDA's
> MSBuild integration to be installed into VS; if it isn't, CMake fails with
> "No CUDA toolset found." Ninja invokes `nvcc` directly and avoids this.

### 4. Run

```
build\srt.exe "path\to\movie.mkv"
```

---

## Packaging a release

```
powershell -ExecutionPolicy Bypass -File scripts\package.ps1 -Version 0.1.0
```

Produces `dist\SRTCreator-v<version>-win64-cuda.zip` containing `srt.exe`, exactly
the FFmpeg / CUDA / VC++ runtime DLLs it needs, this README, and license notices.
Whisper models are **not** bundled (they download on first run).

> The zip is ~460 MB, almost entirely due to NVIDIA's `cublasLt64_*.dll`. That is
> inherent to shipping cuBLAS.

---

## Troubleshooting

- **Empty SRT / 0 segments** — usually means the audio has no detectable speech in
  the processed range, or you used an English-only (`.en`) model on non-English
  audio. Try the default multilingual model, or `-l <code>`.
- **Wrong audio track** — use `--audio-stream <n>` (list tracks with
  `ffprobe -show_streams input`).
- **Model download fails** — check your network, or pre-fetch with
  `scripts\fetch-models.ps1`.

---

## Licensing

SRTCreator's own code, plus bundled components: whisper.cpp/ggml (MIT), FFmpeg
(LGPL v2.1, audio-only, unmodified shared libs), NVIDIA CUDA runtime
(redistributable), and the Microsoft VC++ runtime. See `THIRD_PARTY_NOTICES.txt`
and the `LICENSE-*.txt` files in the release.
