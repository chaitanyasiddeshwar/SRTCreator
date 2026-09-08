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

> Curious how it fits together — the CUDA/whisper, ONNX/DirectML, and FFmpeg
> stacks and every runtime DLL? See [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Features

- **Any input** — MP4, MKV, AVI, TS, MOV, WebM, … with AAC, AC3/E-AC3, DTS,
  TrueHD, Opus, FLAC, MP3, PCM and more (multichannel audio uses dialogue front-center extraction).
- **GPU-accelerated vocal isolation (always-on)** — MDX-Net model (Kim_Vocal_2 via DirectML)
  strips musical score and heavy background sound before transcription.
- **Whisper ASR with Silero VAD (always-on)** — `large-v3-turbo` with flash attention
  and Silero VAD prevents repetition loops and cuts non-speech hallucinations.
- **3-Pass Timing & Quality Pipeline**:
  - **Pass 1: Speech Transcription** — Whisper ASR with acoustic onset/offset tightening and DTW word-level timing.
  - **Pass 2: Silence Sanitization & Timeline Mapping** — Generates a timeline JSON partitioning the audio into alternating `silence` and `vocal` intervals; clamps trailing ends over silence, snaps leading starts to true speech onset, splits cues across mid-sentence pauses (≥ 1.5s), and prunes pure silence hallucinations.
  - **Pass 3: Targeted Audio Infill** — Automatically detects vocal intervals lacking subtitle coverage and re-transcribes them using high-speed, single-session GPU inference to recover missed dialogue.
- **Per-Phase Timing & Summaries** — Detailed runtime logs in `hh:mm:ss` for all 5 phases (Audio extraction, Voice isolation, Transcription, Silence sanitization, Targeted infill) and comprehensive post-processing statistics.
- **Automatic language detection & Translation** (`--translate`).
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
   The first run downloads the speech model (~830 MB), VAD model, and vocal isolation model into
   `%LOCALAPPDATA%\SRTCreator\models`. Subsequent runs start instantly.

The subtitle file is written next to the input (`movie.srt`) along with `<input>.timeline.json` unless you pass
`-o`.

---

## GUI (drag-and-drop)

`srtgui.exe` is a clean, minimal window: **drag a media file onto it** and it decodes,
isolates vocals, transcribes, sanitizes against timeline silence, infills missed dialogue,
and saves `<input>.srt` and `<input>.timeline.json` next to the file.

- **Checkboxes**:
  - **Translate → English** — translate foreign audio to English subtitles.
  - **Flash attention** — fused flash attention CUDA kernels (on by default).
  - **Word timestamps** — DTW word-level acoustic alignment (on by default).
  - **Wrap lines (42)** — Netflix-style character wrap limit.
  - **Center channel (dialogue)** — front-center channel extraction for 5.1/7.1 audio.
  - **Infill speech (Pass 3)** — targeted re-transcription of missed vocal regions (on by default).
  - **Dump audio (debug)** — export the 16 kHz isolated vocal stem (`<name>.vocals16k.wav`).
- **Dropdowns**: **Model** (Whisper), **Language**, and **Vocal** (Kim_Vocal_2 default).
- **Live scrolling transcript panel**: Displays real-time decoding, per-phase timing (`hh:mm:ss`),
  infilled cue alerts, and complete Post-Processing Summary.

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
| `--no-center` | center on | Don't isolate the Front-Center channel. By default, 5.1/7.1 mixes extract the dialogue center channel. |
| `--vocal-model <name>` | Kim_Vocal_2 | Separation model: `Kim_Vocal_2`, `UVR-MDX-NET-Inst_HQ_3`, `UVR_MDXNET_KARA_2`. |
| `--no-infill` | infill on | Disable Pass 3 targeted audio infill for missed vocal regions. |
| `--pause-split <sec>` | `1.5` | Silence duration threshold in seconds to split a mid-sentence cue in Pass 2. |
| `--timeline-json <path>` | `<output>.timeline.json` | Explicit path for the timeline intervals and actions JSON export. |
| `--no-flash-attn` | flash on | Disable flash attention (rarely needed). |
| `--no-word-timestamps` | word-ts on | Disable DTW word-level timing alignment. |
| `--time-offset <sec>` | `0` | Shift every cue by a constant (`+` later, `-` earlier). |
| `--dump-audio [path]` | off | Export the exact 16 kHz isolated vocal audio fed to Whisper as a WAV. |
| `--threads <n>` | auto | CPU worker threads for pre/post-processing. |
| `--models-dir <path>` | `%LOCALAPPDATA%\SRTCreator\models` | Where models are stored/downloaded. |
| `--download <name>` | — | Download a model and exit (e.g. `srt --download large-v3`). |
| `--debug` | off | Write a detailed timing diagnostics report (`<output>.debug.txt`). |
| `--verbose` | off | Print verbose progress and Whisper internals. |
| `-h, --help` | — | Show help. |

### Examples

```bat
:: Basic - vocal isolation, Whisper large-v3-turbo, VAD, 3-pass pipeline
srt "D:\Movies\movie.mkv"

:: Non-English film -> English subtitles
srt "D:\Movies\film.fr.mkv" --translate

:: Quick 2-minute preview
srt movie.mp4 --duration 120 -o preview.srt

:: Adjust mid-sentence pause split threshold to 2.0s
srt movie.mkv --pause-split 2.0

:: Custom output and timeline paths
srt movie.mp4 -o subs\movie.srt --timeline-json subs\analysis.json

:: Nudge subtitles 0.3s later if there is an external delay
srt movie.mkv --time-offset 0.3
```

### Debugging sync & audio

Two flags help when subtitles feel out of sync or the transcription looks wrong:

- **`--dump-audio [path]`** writes the precise 16 kHz mono audio that whisper
  transcribes to a `.wav` (next to `srt.exe` by default). This is exactly what the
  model "hears" — with `--isolate-vocals` it's the isolated vocal stem, so you can
  listen and judge whether separation actually cleaned up the dialogue. It's
  turned on automatically whenever `--isolate-vocals` is used.
- **`--time-offset <sec>`** shifts every cue by a fixed amount. Whisper's
  timestamps aren't perfectly frame-accurate; if you see a consistent lead/lag,
  correct it here (e.g. `--time-offset -0.3` to move cues earlier). To measure it,
  compare against a known-good reference track — extract a film's embedded English
  subtitles with `ffmpeg -i movie.mkv -map 0:s:m:language:eng -c:s srt ref.srt` and
  diff the timings.

Sync fidelity is maintained through the automated 3-pass pipeline:
1. **Pass 1 Acoustic Alignment**: Silero VAD caps speech chunks, and cross-attention DTW pins each cue to spoken word tokens.
2. **Pass 2 Silence Sanitization & Timeline Enforcement**: Subtitles are cross-referenced with the audio timeline. Cues that start prematurely are snapped forward to the true vocal onset, cues lingering into silence are clamped, cues bridging across a silent pause ≥ 1.5s are split into natural sub-phrases, and zero-energy hallucinations are pruned.
3. **Pass 3 Targeted Audio Infill**: Any speech region that was skipped or compressed during the initial transcription pass is re-analyzed and infilled directly using single-session GPU inference.

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
