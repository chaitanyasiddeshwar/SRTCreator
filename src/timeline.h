#pragma once

#include <functional>
#include <string>
#include <vector>

#include "srt.h"
#include "transcribe.h"

namespace timeline {

enum class IntervalType {
    Silence,
    Vocal
};

struct Interval {
    IntervalType type;
    double t0 = 0.0;
    double t1 = 0.0;
    float  avg_db = -90.0f;
    double duration() const { return t1 - t0; }
};

struct Analysis {
    double total_duration = 0.0;
    double total_vocal_sec = 0.0;
    double total_silence_sec = 0.0;
    double vocal_coverage_pct = 0.0;
    int vocal_count = 0;
    int silence_count = 0;
};

struct Action {
    int cue_index = 0;
    std::string action_type; // "clamp_trailing", "snap_leading", "split_pause", "drop_hallucination", "prune_boundary_bleed", "infill_recovered", "reanchor_displaced"
    double orig_t0 = 0.0, orig_t1 = 0.0;
    double new_t0 = 0.0, new_t1 = 0.0;
    std::string text;
    std::string reason;
};

struct MissingRegion {
    double t0 = 0.0;
    double t1 = 0.0;
    float  avg_db = -90.0f;
    double coverage = 0.0; // existing subtitle coverage ratio (0.0 to 1.0)
    bool   force_reanchor = false; // true if flagged by boundary bleed elimination
    double duration() const { return t1 - t0; }
};

struct TimelineMap {
    std::string media_path;
    Analysis analysis;
    std::vector<Interval> intervals;
    std::vector<MissingRegion> missing_vocal;
    std::vector<double> post_bleed_vocal_targets;
    std::vector<Action> actions;
};

// Build continuous alternating silence/vocal intervals covering the entire audio buffer [0, duration].
TimelineMap build_timeline(const std::string& media_path,
                           const std::vector<float>& pcm,
                           const std::vector<transcribe::VadRegion>& vad_regions);

// Pass 2: Detect and eliminate artificial VAD boundary bleeds (cues bunched before long silences).
// Acoustically verifies suspect cues against isolated audio slices using an active Whisper session.
int eliminate_boundary_bleeds(std::vector<srt::Segment>& segments,
                              TimelineMap& map,
                              const std::vector<float>& pcm,
                              transcribe::Session& session,
                              const transcribe::Options& base_opts,
                              std::function<void(const std::string&)> on_log = nullptr);

int eliminate_boundary_bleeds(std::vector<srt::Segment>& segments,
                              TimelineMap& map,
                              const std::vector<float>& pcm,
                              const transcribe::Options& base_opts,
                              std::function<void(const std::string&)> on_log = nullptr);

// Pass 2 (pre-step): Global repetition-loop suppression. Whisper can lock into a
// hallucination loop, emitting one mutated line hundreds of times across the whole
// film (e.g. "Ramin Ares, defender of Master Control." x1700). The windowed SRT
// dedup only catches repeats within a few cues, so a spread-out loop survives AND
// its regions look "covered" to Pass 2.5, hiding the real dialogue underneath.
// This removes every cue whose exact normalized text (>= min_words words) appears
// more than max_repeats times, leaving those vocal regions uncovered so Pass 2.5
// flags them and Pass 3 re-transcribes them in isolation (which decodes cleanly,
// without the VAD-seam that spawned the loop). Returns the number of cues removed.
int suppress_repetition_loops(std::vector<srt::Segment>& segments,
                              TimelineMap& map,
                              int max_repeats = 8,
                              int min_words = 3,
                              std::function<void(const std::string&)> on_log = nullptr);

// Pass 2: Sanitize subtitle cues against the timeline map:
// - Clamps trailing ends that linger across silence
// - Snaps leading starts that start prematurely in silence
// - Splits cues across mid-sentence pauses >= split_pause_threshold (default 1.5s)
// - Prunes hallucination cues that fall 100% in silence
bool sanitize_and_split(std::vector<srt::Segment>& segments,
                        TimelineMap& map,
                        double split_pause_threshold = 1.5);

// Pass 2.5: Detect vocal regions that lack subtitle coverage (potential missed dialogue).
// Also queues post-bleed vocal regions for acoustic re-anchoring.
void find_missing_vocal_regions(const std::vector<srt::Segment>& segments,
                                TimelineMap& map,
                                double min_duration = 0.8,
                                double min_coverage = 0.20);

// Pass 3: Re-transcribe missed and un-anchored vocal regions using targeted Whisper audio slicing.
// Uses an active Whisper session to avoid repeatedly reloading model weights or reinitializing CUDA.
bool infill_missing_regions(const std::vector<float>& pcm,
                            transcribe::Session& session,
                            const transcribe::Options& base_opts,
                            TimelineMap& map,
                            std::vector<srt::Segment>& segments,
                            std::function<void(const std::string&)> on_log = nullptr);

// Standalone fallback (initializes a single Whisper session for all regions).
bool infill_missing_regions(const std::vector<float>& pcm,
                            const transcribe::Options& base_opts,
                            TimelineMap& map,
                            std::vector<srt::Segment>& segments,
                            std::function<void(const std::string&)> on_log = nullptr);

// Serialize the complete timeline, analysis, missing regions, and actions to JSON.
bool write_json(const TimelineMap& map, const std::string& json_path, std::string& err);

struct ActionCounts {
    int clamped = 0;
    int snapped = 0;
    int split = 0;
    int dropped = 0;
    int bleeds_pruned = 0;
    int infilled = 0;
    int reanchored = 0;
    int loops_dropped = 0;
    int total() const { return clamped + snapped + split + dropped + bleeds_pruned + infilled + reanchored + loops_dropped; }
};

ActionCounts count_actions(const TimelineMap& map);

} // namespace timeline
