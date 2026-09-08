#pragma once
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace srt {

// One subtitle span. Times are in seconds.
struct Segment {
    double t0 = 0.0;
    double t1 = 0.0;
    std::string text;
};

// Collapses whisper's repeated hallucination cues. Whisper loops the same line
// over ambiguous/scored audio - sometimes back to back, sometimes alternating
// (A B A B). Feed each cue's text in order; is_duplicate() returns true when the
// cue matches any of the last `window` distinct cues (normalized: lowercased,
// punctuation stripped, whitespace collapsed) and should be dropped. Shared by
// the SRT writer and the GUI's live transcript so both stay clean and identical.
class Deduper {
public:
    explicit Deduper(std::size_t window = 6) : window_(window) {}
    bool is_duplicate(const std::string& text);
private:
    std::size_t window_;
    std::deque<std::string> recent_; // normalized text of recently kept cues
};

// "HH:MM:SS,mmm"
std::string format_timestamp(double seconds);

// Word-wrap text at word boundaries so no line exceeds max_width characters,
// balancing into two lines when it fits in two. max_width <= 0 disables wrapping.
std::string wrap(const std::string& text, int max_width);

// Write segments as UTF-8 SRT. Each segment's text is wrapped to max_line_length
// (<= 0 disables). Empty-text segments are skipped and indices are assigned to
// the emitted lines only. Returns false (and sets err) on I/O error.
bool write(const std::vector<Segment>& segs, const std::string& path,
           int max_line_length, std::string& err);

// Parse an SRT file into segments (times in seconds; newlines in text become
// spaces; markup like <i> and {\an8} is stripped). For loading a reference track
// to compare timing against. Returns false (and sets err) on I/O error.
bool read(const std::string& path, std::vector<Segment>& out, std::string& err);

} // namespace srt
