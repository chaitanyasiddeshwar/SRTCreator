# SRTCreator quality baseline

Purpose: a **regression guard**. Before/after any change, regenerate the Tron SRT
with all options on and re-run the harness; the composite score and the
word-coverage numbers must not drop.

## Current baseline: after global loop-suppression fix (2026-09-08)

`timeline::suppress_repetition_loops()` added + wired before Pass 2.5 (uncommitted).

| Metric | Pre-fix (f2511c4) | **After loop-suppress fix** |
|---|---|---|
| Composite | 29.13 | **83.56** |
| Well-timed words (<=0.5s) | 18.85% | **82.51%** |
| Present words (<=6s) | 20.93% | **84.65%** |
| Truly missing | 79.07% | **15.35%** |
| Mistimed-only | 2.08% | 2.14% |
| Cue recall / precision | 16.6% / 74.7% | **72.9% / 83.7%** |
| Cue timing within 1s | 83.9% | **86.0%** |
| Final cues | 241 | **944** |
| Loop cues in output | 34 | **0** |

Job: suppressed **2014** loop cues → Pass 3 re-infilled **933** recovered cues.
Spot-check of a formerly-empty region (00:08:26+) recovered near-verbatim dialogue
at ~0.5-1s timing. Artifacts: `baseline/tron_loopfix.*`. **Do not regress below
composite 83.56 / well-timed 82.5% / truly-missing 15.4%.**

### Cross-validation on Transformers.One (no-regression check)

Transformers.One was already clean (no severe loop). Running the fix build on it:
- removed **21** cues (it had its own minor hidden loop, "i'm not in the middle of
  the game" x11, that the windowed dedup missed) and re-infilled them;
- output is **99.83% well-timed / 99.75% recall / 99.81% precision identical** to the
  prior good output (4 missing, 3 extra of 1596).

Conclusion: **big win on the pathological file, no regression (small tidy) on the
clean one.** The suppressor generalizes and is safe. (No embedded subtitle exists in
the Transformers mp4, so its prior generated SRT is the regression reference, not a
ground truth.)

---

## Historical: pre-fix baseline (commit `f2511c4`, build 2026-09-08 17:08)

## How to reproduce

```sh
# 1. generate (all options on: center channel, vocal isolation, Pass3 infill, debug)
build/srt.exe "D:\MediaServer\Tron.Ares.2025.2160p.UHD.BluRay.HDR.DoVi.TrueHD 7.1.Atmos.x265-SPHD.mkv" \
  -o out.srt --debug \
  --reference "D:\MediaServer\Tron.Ares.2025.2160p.UHD.BluRay.HDR.DoVi.TrueHD 7.1.Atmos.x265-SPHD.real.srt"

# 2. score it
python scripts/compare_srt.py out.srt \
  "D:\MediaServer\Tron.Ares.2025.2160p.UHD.BluRay.HDR.DoVi.TrueHD 7.1.Atmos.x265-SPHD.real.srt"
```

The harness (`scripts/compare_srt.py`) is deterministic and stdlib-only. It scores
**100.0** on reference-vs-itself, so the metric itself is sound.

## Metric definitions

- **Word-level coverage (segmentation-independent)** — the headline metric,
  immune to how cues are merged/split:
  - *well-timed* (`strict`, ≤0.5 s): ref words present in a gen cue near the right time.
  - *present* (`lenient`, ≤6 s): ref words present within 6 s.
  - *truly missing* = 100 − present (dialogue never transcribed near its time).
  - *mistimed only* = present − well-timed (transcribed but time is wrong).
- **Cue-level** recall/precision/timing — sensitive to merging; informational only.
- **Composite** = 0.60·well-timed + 0.25·(100 − truly-missing) + 0.15·within-1s.

## Baseline numbers

**Determinism (measured):** two back-to-back full runs produced **byte-for-byte
identical** SRTs (MD5 `d8c13ab68dbf8770f4965c6c940db22f`). So on this machine the
pipeline is reproducible at cue granularity and these are **hard numbers**, not a
band. (CLAUDE.md §14 warns of CUDA run-to-run variance; it did not manifest here —
any variance is sub-threshold and absorbed before cue output.) A regression is
therefore any drop below these values, full stop.

| Run | Composite | Well-timed words | Truly missing | Cue recall | Cue within-1s | Gen cues (raw→dedup) |
|-----|-----------|------------------|---------------|-----------|---------------|----------------------|
| 1   | **29.13** | 18.85%           | 79.07%        | 16.61%    | 83.89%        | 1913 → 241           |
| 2   | **29.13** | 18.85%           | 79.07%        | 16.61%    | 83.89%        | 1913 → 241 (identical) |

Reference: 1084 cues. Movie runtime ~1:53. Total job time: **11:44**
(audio extraction 6:58, isolation 2:39, transcription 1:26, Pass2 0:22, Pass3 0:19).

## Dominant finding (run 1): catastrophic repetition-loop hallucination

The raw whisper output (1913 cues) is **~89% one looped hallucination**:

- `"Ramin Ares, defender of Master Control."` — **1029×**
- `"Ramin Ares, defender of our Master Control."` — **673×**
- = **~1702 / 1913 cues**.

The real dialogue is *"Program name, Ares. Defender of the Grid."* (ref cue at
~line 196). Whisper mutated it and looped it, first appearing at **00:05:38** and
recurring across the **entire** film.

Consequences:
- The windowed dedup (`srt.cpp`, window 6) collapses the loop to 241 surviving
  cues, but the loop is spread out (>6 cues apart), so **34 loop cues still leak
  into the final SRT**.
- More importantly, **while whisper loops it is not decoding the real speech**, so
  that dialogue is lost → 79% truly-missing.
- Timing of what *is* captured is actually fine (cue within-1s 83.9%, median
  |Δ| 0.22 s). **The primary problem is missing dialogue from the loop, not
  mis-timing.**

Global word recall ignoring time is 76% (inflated by common words), confirming the
transcriber *can* hear most words — the loop is what suppresses coherent,
correctly-placed dialogue.

### Direction (per user, 2026-09-08)
Isolation is confirmed good (audible in the dumped WAV); **all issues are whisper
hallucinations**, so fixes belong in the transcribe/whisper layer, not separation.
The model used was the recommended `large-v3-turbo-q8_0` (not large-v3), so this is
not a wrong-model artifact. `no_context=true`, `suppress_nst=true`, temperature
fallback, and `max_speech_duration_s=15` are all already on, yet the loop forms.

Two complementary, whisper-focused fix candidates (not yet implemented):
1. **In-decode guards** — tune `entropy_thold` / `logprob_thold` / `no_speech_thold`
   and temperature fallback so a low-entropy repeat trips fallback before a window
   locks into the loop.
2. **Global loop suppression pass** — the current dedup (`srt.cpp`) is *windowed*
   (6 cues), so a line repeated ~1700× but spread out survives. Add a global
   frequency check ("normalized line appears > ~8× across the whole film → loop
   hallucination"): drop all instances and mark those regions `missing_vocal` so
   Pass 3 re-infills them in isolation (which §3.1 says decodes cleanly). Pass 3
   already does this for post-bleed regions; this widens the net.

Any implementation must be validated by re-running the reproduce steps and showing
the composite score **rises above 29.13** with truly-missing **below 79%**.

## Files
- `tron_run1.srt` — generated SRT (run 1)
- `tron_run1.debug.txt` — per-cue debug report
- `tron_run1.metrics.json` — full harness output incl. missing-cue list
- `tron_run1.run.log` — full pipeline log
- `../scripts/compare_srt.py` — the scoring harness
