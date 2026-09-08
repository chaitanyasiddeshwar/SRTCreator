#pragma once
#include <string>
#include <vector>
#include "srt.h"
#include "transcribe.h"

// Timing diagnostics. Writes a human-readable report correlating three views of
// the same timeline so we can see WHY a cue appears early:
//   1. what VAD produced   (speech regions it kept),
//   2. what whisper output  (the cues and their times),
//   3. what the audio holds  (per-bin energy / silence of the 16 kHz mono buffer
//      whisper actually heard - the isolated vocal stem in isolation mode),
//   + an optional reference SRT (e.g. the film's embedded subs) as ground truth.
namespace debugreport {

struct Info {
    std::string model;
    bool isolate = false, center = false, vad = false, word_ts = false;
};

// Write the report to `path`. `pcm16k` is the exact buffer fed to whisper;
// `cues` the final cues; `vad` the VAD regions (may be empty); `reference` an
// optional ground-truth track (nullptr if none). Returns false/err on I/O error.
bool write(const std::string& path,
           const std::vector<float>& pcm16k,
           const std::vector<srt::Segment>& cues,
           const std::vector<transcribe::VadRegion>& vad,
           const std::vector<srt::Segment>* reference,
           const Info& info, std::string& err);

} // namespace debugreport
