#pragma once
#include <string>
#include <vector>

namespace srt {

// One subtitle span. Times are in seconds.
struct Segment {
    double t0 = 0.0;
    double t1 = 0.0;
    std::string text;
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

} // namespace srt
