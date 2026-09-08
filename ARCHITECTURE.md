# SRTCreator — Architecture

End-to-end map of how the code, the two inference engines, and every runtime DLL
fit together. Everything here is verified against the actual build
(`dumpbin /dependents`, `CMakeLists.txt`, and the staged `build\` folder), not
inferred.

> TL;DR: one static core library (`srtcore`) links **whisper.cpp + ggml (CUDA)
> statically into the exe**, and dynamically imports **FFmpeg** (audio decode) and
> **ONNX Runtime + DirectML** (vocal isolation). The exe is ~64 MB because the
> CUDA whisper kernels are baked in. The only *un-shipped* runtime requirement is
> the **NVIDIA driver** (`nvcuda.dll`); the CUDA runtime, cuBLAS, DirectML and
> FFmpeg DLLs all ship next to the exe.

---

## 1. Build artifacts

Two executables, one shared static library — the logic is **not** duplicated.

```
                         ┌─────────────────────────────────────────┐
                         │  srtcore.lib  (STATIC)                   │
   src/main.cpp  ───────▶│    audio.cpp       transcribe.cpp       │◀─────── src/gui/main_gui.cpp
   (CONSOLE subsystem)   │    separate.cpp    timeline.cpp         │        (WINDOWS subsystem)
        │                │    srt.cpp         models.cpp           │              │
        ▼                │    log.cpp         debug.cpp            │              ▼
     srt.exe             └─────────────────────────────────────────┘          srtgui.exe
   CLI front-end                      (all the pipeline)                    Win32 GUI front-end
```

- `srt.exe` — `add_executable(srt src/main.cpp)`, console subsystem.
- `srtgui.exe` — `add_executable(srtgui WIN32 …)`, windows subsystem (+ `comctl32`).
- Both link `srtcore` (`CMakeLists.txt:71-75`). See `CLAUDE.md` §11 and
  `ARCHITECTURE §7` for why two subsystems means two exes.

---

## 2. Source modules

| Module | Responsibility | Talks to (external) |
|---|---|---|
| `main.cpp` | CLI parsing, orchestration, `--time-offset`, `--dump-audio` | — |
| `gui/main_gui.cpp` | Win32 window, drag-drop, live transcript, worker thread, `JobCleanup` RAII guard | `comctl32`, `user32`, `gdi32`, `shell32`, `comdlg32` |
| `audio.{h,cpp}` | Demux/decode/resample any container → 16 kHz mono f32; center-channel matrix; WAV dump | **FFmpeg**: `avformat`, `avcodec`, `avutil`, `swresample` |
| `separate.{h,cpp}` | MDX-Net vocal isolation: STFT → ONNX inference → iSTFT, chunk overlap/trim | **ONNX Runtime (DirectML)**, `pocketfft` (header-only) |
| `transcribe.{h,cpp}` | whisper.cpp wrapper: `Session` (GPU model reuse), params, VAD, flash-attn, DTW word timing, callbacks | **whisper.cpp → ggml → CUDA** (static) |
| `timeline.{h,cpp}` | Timeline analysis, silence sanitization (clamping, snapping, pause-splitting), missing vocal gap detection, JSON export, infill orchestration | — (pure) |
| `srt.{h,cpp}` | SRT formatting, line-wrap, windowed hallucination de-dup | — (pure) |
| `models.{h,cpp}` | Resolve/download whisper, VAD, and MDX models | `urlmon` (`URLDownloadToFile`) |
| `log.{h,cpp}` | Thread-safe file log + crash backtrace | `dbghelp` |
| `debug.{h,cpp}` | Diagnostics logging, timeline dumps, and validation reports | — |

The core is deliberately layered so the CLI and GUI are thin: both just build an
`audio::DecodeOptions` / `transcribe::Options`, call the same functions, and write
with `srt::write`.

---

## 3. Runtime pipeline (data flow)

The pipeline executes in 5 sequential phases with strict resource lifetimes and per-phase timing (`hh:mm:ss`):

```
  input file (any container/codec)
        │
        │  [Phase 1: Audio Extraction] (audio.cpp via FFmpeg)
        ▼
   44.1 kHz stereo / 5.1 / 7.1 audio (center-channel extracted for multichannel)
        │
        │  [Phase 2: Voice Isolation] (separate.cpp via MDX-Net ONNX / DirectML)
        ▼
   pocketfft STFT → ONNX Runtime (DirectML, GPU) → iSTFT (Kim_Vocal_2)
        │  *mix buffers deallocated immediately via .swap()*
        ▼
   resample → 16 kHz mono f32 PCM
        │  *44.1k vocals buffer deallocated immediately via .swap()*
        │  (optional --dump-audio writes *.vocals16k.wav)
        │
        ▼
   [Phase 3: Pass 1 Primary Transcription] (transcribe.cpp via whisper.cpp / CUDA)
        │  Single GPU Session initialized once (`transcribe::Session`)
        ├─▶ whisper_full() [Silero VAD + flash-attn + DTW token timestamps]
        ▼
   Raw Cues (t0/t1/text) + Segment VAD Timestamps
        │
        │  [Phase 4: Pass 2 Silence Sanitization & Timeline Mapping] (timeline.cpp)
        ▼
   Construct partition: alternating [silence] and [vocal] intervals
        ├─▶ Export <input>.timeline.json (intervals + sanitization actions)
        ├─▶ Trailing silence clamping (clamp cue ends extending into silence gaps)
        ├─▶ Leading silence snapping (snap cue starts forward to speech onsets)
        ├─▶ In-sentence pause splitting (split cues spanning silence gaps >= 1.5s)
        ├─▶ Hallucination pruning (drop cues falling 100% inside silence)
        ▼
   [Phase 4.5: Non-Silence Gap Analysis]
        ▼
   Detect uncaptioned speech (vocal intervals >= 0.8s with < 20% subtitle coverage)
        │  Yields missing_vocal_regions list
        │
        ▼
   [Phase 5: Pass 3 Targeted Audio Infill] (timeline.cpp + transcribe::Session)
        │  Targeted re-transcription over missing vocal audio slices
        │  *Reuses active GPU Session (zero model reload / ~25ms per slice)*
        │  *Audio slices freed immediately per chunk*
        ▼
   Merge recovered dialogue cues chronologically
        │  *16 kHz PCM buffer freed immediately via .swap()*
        │  *transcribe::Session closed -> whisper context destroyed -> cudaDeviceReset()*
        │  *Working set compacted via SetProcessWorkingSetSize*
        │
        ▼
   srt::write  ── wrap (42 cols) + windowed de-dup → CRLF UTF-8
        │
        ▼
   <input>.srt  +  <input>.timeline.json
```

Note: **Silero VAD and vocal isolation are always enabled by default.** Running VAD on the isolated stem trims leading silence so cue starts track the real speech onset and eliminates multi-minute repetition loops.

---

## 4. Engine A — Whisper ASR on CUDA (statically linked)

This is where the exe's 64 MB comes from. Nothing here is a separate whisper DLL;
it is all compiled into `srt.exe` / `srtgui.exe`.

```
  transcribe.cpp
      │ calls whisper_full()
      ▼
  whisper.lib ─┐
  ggml.lib     │  (all STATIC .lib archives, baked into the exe;
  ggml-base.lib│   BUILD_SHARED_LIBS=OFF in CMakeLists.txt)
  ggml-cpu.lib │
  ggml-cuda.lib┘  ← SM 8.6 (Ampere/3080 Ti) kernels compiled by nvcc, embedded
      │
      │ ggml-cpu uses OpenMP for CPU threads ───────────▶ VCOMP140.dll  (shipped, VC++ redist)
      │
      │ ggml-cuda issues GEMMs via cuBLAS ──────────────▶ cublas64_13.dll   (shipped)
      │                                                        │
      │                                                        ▼
      │                                                  cublasLt64_13.dll  (shipped, 456 MB)
      │
      │ CUDA runtime services (cudart) ─────────────────▶ cudart64_13.dll   (shipped)
      │
      ▼
  CUDA Driver API at runtime ────────────────────────────▶ nvcuda.dll
                                                            (NVIDIA GPU driver,
                                                             System32 — NOT shipped)
```

Key facts (verified):

- `srt.exe` **directly imports `cublas64_13.dll`** (the GEMM path); `cublas64`
  pulls in `cublasLt64_13.dll`. `cudart64_13.dll` is part of the shipped CUDA
  runtime set required by the toolkit libraries.
- The **CUDA *kernels* are static** in the exe (that's the 64 MB), but the CUDA
  **runtime libraries** (`cudart`/`cublas`/`cublasLt`) are DLLs shipped alongside.
- `nvcuda.dll` is the **CUDA Driver API**, installed by the NVIDIA GPU driver and
  loaded from `System32` at runtime. This is the single hard dependency we do
  **not** ship — a recent driver satisfies it.
- Tuning knobs (`transcribe.cpp`): `flash_attn`, Silero `vad` (+ capped
  `max_speech_duration_s`), `suppress_nst`, greedy sampling, half CPU threads for
  pre/post.

### 4.1 Single-Session GPU Model Lifecycle (`transcribe::Session`)

ASR inference across both primary transcription and targeted infill is managed via `transcribe::Session`:
- **Single Context Allocation**: `Session::init(model_path, use_gpu, flash_attn)` loads the Whisper model weights into GPU VRAM once.
- **Cross-Pass GPU Model Reuse**: Pass 1 executes the primary transcription on the full audio. When Pass 3 infills uncaptioned speech, it runs `Session::run_slice()` on each missing vocal region using the **already-resident GPU model**. This completely eliminates multi-GB model reloads from disk and avoids recreating CUDA runtime contexts hundreds of times (which previously saturated Windows file cache, exhausted pinned host memory, and caused GPU driver lockups / machine crashes).
- **Sub-Second Targeted Inference**: Individual missing vocal slices (~1–4 seconds each) transcribe in ~25 ms directly on the GPU without context re-initialization overhead.
- **Explicit Cleanup**: `Session::close()` frees the `whisper_context`, followed immediately by `cudaDeviceReset()` to release all stranded driver memory before returning to the UI or shell.

---

## 5. Engine B — Vocal isolation on DirectML (dynamic)

Vocal isolation is enabled by default (Kim_Vocal_2) to strip soundtrack score before
transcription. Its DLLs are a **load-time** dependency of the exe (see
§6), so `onnxruntime.dll` + `DirectML.dll` must be present.

```
  separate.cpp
      │ STFT (pocketfft, header-only, compiled in)
      ▼
  Ort::Session on the DirectML EP
      │  OrtSessionOptionsAppendExecutionProvider_DML(so, 0)
      ▼
  onnxruntime.dll  (shipped, 16.5 MB)
      │  runs the MDX-Net .onnx model
      ▼
  DirectML.dll     (shipped, 17.7 MB)
      │  DirectML compute
      ▼
  d3d12.dll, dxgi.dll   (Windows system; GPU vendor's D3D12 driver does the math)
      │
      ▼
  vocals spectrogram → separate.cpp iSTFT → mono stem
```

Why DirectML and not the ONNX **CUDA** EP? Deliberate decoupling: the CUDA EP
would pin ONNX Runtime to a specific CUDA/cuDNN version, which would collide with
our CUDA-13 whisper build. DirectML rides the vendor D3D12 driver instead, so the
two GPU stacks never fight over CUDA/cuDNN versions. (See `CLAUDE.md` §12.)

MDX model params live in a registry in `separate.cpp` (Kim_Vocal_2 default,
Inst_HQ_3, KARA_2), from UVR's `model_data.json`.

---

## 6. Complete DLL dependency map

Grouped by where each DLL comes from. "Direct" = named in the exe's PE import
table (verified with `dumpbin /dependents`); "transitive" = pulled in by another
shipped DLL; "system" = provided by Windows / the GPU driver.

### Shipped next to the exe (in `build\`, and in the release zip)

| DLL | Size | Pulled in by | Purpose |
|---|---:|---|---|
| `cublasLt64_13.dll` | 456 MB | `cublas64_13.dll` | cuBLAS heuristics (dominates zip size) |
| `avcodec-63.dll` | 68 MB | **srt.exe direct** | audio decoders |
| `cublas64_13.dll` | 48 MB | **srt.exe direct** | GEMM for ggml-cuda |
| `avfilter-12.dll` | 29 MB | *(unused — see note)* | FFmpeg filters |
| `avformat-63.dll` | 21 MB | **srt.exe direct** | demuxers/containers |
| `DirectML.dll` | 18 MB | `onnxruntime.dll` | GPU compute for ONNX |
| `onnxruntime.dll` | 17 MB | **srt.exe direct** | MDX-Net inference |
| `avdevice-63.dll` | 3.7 MB | *(unused)* | FFmpeg devices |
| `avutil-61.dll` | 2.8 MB | **srt.exe direct** | FFmpeg core utils |
| `swscale-10.dll` | 2.4 MB | *(unused — video)* | FFmpeg scaling |
| `swresample-7.dll` | 0.7 MB | **srt.exe direct** | resample to 16 kHz mono |
| `cudart64_13.dll` | 0.45 MB | CUDA runtime set | CUDA runtime services |

> **Unused FFmpeg DLLs**: `stage_runtime_dlls()` in `CMakeLists.txt` globs *all*
> `third_party/ffmpeg/bin/*.dll`, so `avfilter`, `avdevice`, and `swscale` are
> copied even though the audio path never calls them. The eventual minimized
> static FFmpeg build (`scripts/build-ffmpeg.sh`, `CLAUDE.md` §4.3) removes these.

### System — Windows itself (present on any Win10/11)

- `KERNEL32`, `ADVAPI32`, `USER32`, `GDI32`, `SHELL32`, `COMDLG32`, `COMCTL32`
  (GUI only), `urlmon.dll` (model downloads), `dbghelp.dll` (crash traces).
- `d3d12.dll`, `dxgi.dll` — Direct3D 12, used by DirectML (the GPU vendor's D3D12
  user-mode driver does the actual work).
- The `api-ms-win-crt-*` and `api-ms-win-core-*` "API set" stubs — the Universal
  CRT and Win32 API sets that ship with the OS.

### System — NVIDIA GPU driver (the one thing we do NOT ship)

- `nvcuda.dll` — CUDA Driver API. Installed with the GPU driver, loaded from
  `System32`. Required for whisper's CUDA backend to initialize.

### Microsoft VC++ runtime (bundled in the release zip)

- `MSVCP140.dll`, `MSVCP140_1.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`,
  `VCOMP140.dll` (OpenMP). Imported by the exe and by `onnxruntime.dll`.

### Load order at startup (both exes)

```
srt(gui).exe
 ├─ cublas64_13.dll ── cublasLt64_13.dll
 ├─ avformat-63.dll ┬─ avcodec-63.dll ─ avutil-61.dll
 │                  └─ swresample-7.dll ─ avutil-61.dll
 ├─ onnxruntime.dll ─ DirectML.dll ─ d3d12.dll, dxgi.dll
 ├─ urlmon.dll, dbghelp.dll
 ├─ VCOMP140 / MSVCP140 / VCRUNTIME140[_1]  (VC++ redist)
 ├─ api-ms-win-crt-*  (Universal CRT — OS)
 └─ KERNEL32 / ADVAPI32 [ + USER32/GDI32/SHELL32/COMDLG32/COMCTL32 for the GUI ]

  ...and at CUDA init, dynamically: nvcuda.dll (from the installed GPU driver)
```

> Because `onnxruntime.dll`, `DirectML.dll`, and `cublas64_13.dll` are **implicit
> (load-time) imports**, all three must be present for the exe to *start at all* —
> even for a plain CLI transcription that uses neither isolation nor, in that run,
> much cuBLAS. Deleting them yields a "DLL not found" failure at launch, not a
> graceful fallback.

---

## 7. Models on disk (data, not code)

Downloaded on first use (`models.cpp`, `separate.cpp`) into
`%LOCALAPPDATA%\SRTCreator\models` (override with `--models-dir`):

| File | Engine | Notes |
|---|---|---|
| `ggml-large-v3-turbo-q8_0.bin` | whisper | default ASR model (~830 MB) |
| other `ggml-*.bin` | whisper | on demand (`-m` / `--download`) |
| `ggml-silero-v*.bin` | VAD | Silero VAD for the default path |
| `Kim_Vocal_2.onnx` etc. | MDX-Net | vocal isolation (opt-in) |

Models are **not** in the repo or the release zip — they fetch at runtime via
`urlmon`.

---

## 8. Distribution & runtime requirements

Shipped as a folder/zip (`scripts/package.ps1`): the exe(s), the shipped DLLs from
§6, README/licenses. Models download on first run.

**End-user needs only:** a reasonably recent **NVIDIA GPU driver** (for
`nvcuda.dll`). The CUDA runtime, cuBLAS, DirectML, FFmpeg, and VC++ runtime DLLs
are all in the zip — no CUDA Toolkit or VC++ redistributable install required.

- The zip is ~460 MB, almost entirely `cublasLt64_13.dll`.
- DirectML rides whatever D3D12-capable GPU/driver is present; it does not require
  NVIDIA specifically (only whisper's CUDA path does).

---

## 9. Two GPU stacks, on purpose

| Stage | Framework | GPU API | Linkage |
|---|---|---|---|
| Whisper ASR | ggml-cuda | **CUDA 13** (cuBLAS + driver) | static kernels in exe + shipped runtime DLLs |
| Vocal isolation | ONNX Runtime | **DirectML** (D3D12) | shipped `onnxruntime.dll` + `DirectML.dll` |

Keeping isolation on DirectML instead of the ONNX CUDA EP is the deliberate
decision that lets the whisper CUDA build and the ONNX build coexist without
CUDA/cuDNN version coupling. The cost is two GPU runtimes in the package; the
benefit is that upgrading one engine never breaks the other.

---

## 10. Multi-Pass Pipeline Architecture (Pass 1, 2 & 3)

To ensure **no subtitles linger on screen during silence** and **no spoken dialogue is missed**, the pipeline uses a multi-pass architecture:

```
  16 kHz mono audio buffer (PCM)
        │
        ▼
  [Pass 1: Primary Transcription & Timeline Generation]
        │
        ├─▶ whisper_full() (CUDA, flash-attn, word-timestamps) ──▶ Raw Segments
        │
        └─▶ Silero VAD + Acoustic Energy Partitioning ──────────▶ <input>.timeline.json
                                                                  (complete partition of vocal
                                                                   and silence intervals)
        │
        ▼
  [Pass 2: Silence Sanitization & In-Sentence Pause Splitting]
        │
        ├─▶ Trailing silence clamping: end time clamped to vocal offset + read buffer
        ├─▶ Leading silence snapping: start time snapped forward to speech onset
        ├─▶ In-sentence pause splitting: if a cue spans a silence gap >= 1.5s,
        │   split the line into two cues so screen is clear during silence
        ├─▶ Hallucination pruning: drop cues with 100% overlap with pure silence
        │
        ▼
  [Pass 2.5: Non-Silence Gap Analysis]
        │
        └─▶ Scan all vocal intervals against subtitle coverage
            Uncaptioned vocal intervals (>= 0.8s with no subtitle) ──▶ missing_vocal_regions
        │
        ▼
  [Pass 3: Targeted Audio Infill]
        │
        ├─▶ Extract audio slices for missing vocal regions (0.35s padding)
        ├─▶ Targeted whisper decoding on missed speech chunks via transcribe::Session
        └─▶ Chronologically insert recovered cues into subtitle stream
        │
        ▼
   srt::write  ── wrap (42 cols) + windowed de-dup ──▶ <input>.srt
```

### 10.1 Pass 1: Primary Transcription & Timeline Generation
Whisper ASR executes on the full 16 kHz audio buffer using Silero VAD and DTW token timestamps (`word_timestamps = true`). In parallel, acoustic RMS energy and Silero VAD segments are combined to generate an exact partition of alternating `vocal` and `silence` intervals across the entire duration.

### 10.2 Pass 2: Silence Sanitization & In-Sentence Pause Splitting
Subtitles are validated and aligned against the timeline partition:
- **Trailing Silence Clamping**: Prevents subtitles from lingering over silence when characters stop talking. The cue end is clamped to the vocal boundary plus an adaptive reading buffer ($\le 1.4\text{s}$).
- **Leading Silence Snapping**: Snaps back-dated cue starts forward to the true acoustic onset of speech.
- **In-Sentence Pause Splitting**: If an actor pauses for $\ge 1.5\text{s}$ mid-sentence, the single subtitle cue is cleanly split at the silence boundary across sentence punctuation/words. The screen remains completely clear during the pause.
- **Hallucination Pruning**: Cues that fall entirely inside silence intervals (e.g. repetition loops or instrumental score hallucinations) are removed.
- **Timeline Export**: The complete set of intervals and sanitization actions is exported to `<input>.timeline.json`.

### 10.3 Pass 2.5: Non-Silence Gap Analysis
Every vocal interval is checked against subtitle coverage. Any vocal interval $\ge 0.8\text{s}$ with $< 20\%$ subtitle coverage is identified as an uncaptioned dialogue candidate and queued for infill.

### 10.4 Pass 3: Targeted Audio Infill
The active `transcribe::Session` executes fast, targeted GPU inference over each missing vocal slice (padded by 0.35s). Recovered dialogue cues are re-aligned to global timestamps and chronologically spliced into the subtitle stream.

---

## 11. Memory Management & Resource Lifecycle

Processing high-resolution multi-channel movie audio with deep neural networks requires strict memory budgeting across both host RAM and GPU VRAM.

### 11.1 Audio Buffer Deallocation Hierarchy
Audio buffers are deallocated as early as possible via vector swapping (`std::vector<float>().swap(...)`) rather than waiting for process exit:
1. **Mix / Multichannel Buffer**: Decoded 44.1 kHz stereo (~1.2 GB for 2 hours) is freed immediately after MDX-Net vocal separation finishes.
2. **Vocals Buffer**: Separated 44.1 kHz mono vocals (~600 MB) is freed immediately after resampling to 16 kHz.
3. **Infill Slices**: In Pass 3, each missing vocal slice buffer (~100 KB - 1 MB) is scoped to its iteration and destructed immediately.
4. **16 kHz PCM Buffer**: The 16 kHz f32 audio buffer (~576 MB for 2 hours) is explicitly deallocated via `.swap()` before the SRT file is finalized.

### 11.2 GPU Model & CUDA Context Lifecycle
- **Zero-Reload Model Reuse**: `transcribe::Session` maintains the loaded Whisper model in GPU memory across both Pass 1 and Pass 3. This eliminates disk cache saturation and pinned host memory exhaustion caused by reloading multi-GB weights inside the infill loop.
- **Explicit CUDA Reset**: Upon completion (or failure), `Session::close()` calls `whisper_free(ctx)` followed by `cudaDeviceReset()`. This flushes driver queues and unmaps all device allocations.
- **Working Set Compaction**: Windows Memory Manager retains deallocated heap pages in the process working set by default. SRTCreator explicitly calls:
  ```cpp
  SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
  ```
  forcing Windows to release uncommitted and unreferenced physical pages back to the operating system immediately.
- **RAII GUI Guard**: In `srtgui.exe`, a `JobCleanup` RAII guard guarantees that buffer releases, `Session::close()`, `cudaDeviceReset()`, and working set trimming execute under all exit paths (normal completion, user cancellation, or unexpected errors).
