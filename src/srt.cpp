#include "srt.h"
#include "log.h"
#include <cctype>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

namespace srt {

namespace {

// Normalize a cue for duplicate detection: lowercase, drop punctuation, and
// collapse runs of whitespace. Whisper's hallucination loops re-emit the same
// line with trivial punctuation/case/spacing differences, so comparing the
// normalized form catches far more of them than an exact string match.
std::string normalize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool pending_space = false;
    for (unsigned char c : s) {
        if (std::isspace(c)) { pending_space = !out.empty(); continue; }
        if (std::ispunct(c)) continue; // ',' '.' '-' '<i>' tags etc.
        if (pending_space) { out.push_back(' '); pending_space = false; }
        out.push_back((char)std::tolower(c));
    }
    return out;
}

std::vector<std::string> split_words(const std::string& text) {
    std::vector<std::string> words;
    std::string w;
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (!w.empty()) { words.push_back(w); w.clear(); }
        } else {
            w.push_back(c);
        }
    }
    if (!w.empty()) words.push_back(w);
    return words;
}

std::string join(const std::vector<std::string>& w, size_t from, size_t to) {
    std::string s;
    for (size_t i = from; i < to; ++i) { if (i > from) s.push_back(' '); s += w[i]; }
    return s;
}

// Greedy wrap into as many lines as needed, each <= width.
std::string greedy(const std::vector<std::string>& words, int width) {
    std::string out;
    size_t line_len = 0;
    for (size_t i = 0; i < words.size(); ++i) {
        size_t wl = words[i].size();
        if (line_len == 0) { out += words[i]; line_len = wl; }
        else if ((int)(line_len + 1 + wl) <= width) { out += ' '; out += words[i]; line_len += 1 + wl; }
        else { out += '\n'; out += words[i]; line_len = wl; }
    }
    return out;
}

} // namespace

bool Deduper::is_duplicate(const std::string& text) {
    std::string norm = normalize(text);
    if (norm.empty()) return false; // never drop (empty cues are skipped elsewhere)
    for (const auto& r : recent_) if (r == norm) return true;
    recent_.push_back(std::move(norm));
    if (recent_.size() > window_) recent_.pop_front();
    return false;
}

std::string wrap(const std::string& text, int max_width) {
    std::vector<std::string> words = split_words(text);
    if (words.empty()) return std::string();

    size_t total = words.size() - 1;
    for (const auto& w : words) total += w.size();

    if (max_width <= 0 || (int)total <= max_width) return join(words, 0, words.size());

    // Prefer a balanced two-line layout when the text fits in two lines.
    if ((int)total <= 2 * max_width) {
        int best_k = -1, best_metric = 0;
        for (size_t k = 1; k < words.size(); ++k) {
            std::string l1 = join(words, 0, k);
            std::string l2 = join(words, k, words.size());
            if ((int)l1.size() > max_width || (int)l2.size() > max_width) continue;
            int metric = (int)(l1.size() > l2.size() ? l1.size() : l2.size());
            if (best_k < 0 || metric < best_metric) { best_k = (int)k; best_metric = metric; }
        }
        if (best_k > 0)
            return join(words, 0, best_k) + "\n" + join(words, best_k, words.size());
    }

    // Otherwise wrap greedily across as many lines as needed.
    return greedy(words, max_width);
}


std::string format_timestamp(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    long long ms = (long long)(seconds * 1000.0 + 0.5);
    long long h = ms / 3600000; ms %= 3600000;
    long long m = ms / 60000;   ms %= 60000;
    long long s = ms / 1000;    ms %= 1000;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld,%03lld", h, m, s, ms);
    return std::string(buf);
}

namespace {

// Parse "HH:MM:SS,mmm" (or with '.') to seconds; -1 on failure.
double parse_ts(const std::string& s) {
    int h, m, sec, ms;
    if (std::sscanf(s.c_str(), "%d:%d:%d,%d", &h, &m, &sec, &ms) == 4 ||
        std::sscanf(s.c_str(), "%d:%d:%d.%d", &h, &m, &sec, &ms) == 4)
        return h * 3600.0 + m * 60.0 + sec + ms / 1000.0;
    return -1.0;
}

// Strip SRT/ASS markup (<i>...</i>, {\an8}, etc.) for text comparison.
std::string strip_markup(const std::string& in) {
    std::string out;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '<') { while (i < in.size() && in[i] != '>') ++i; continue; }
        if (in[i] == '{') { while (i < in.size() && in[i] != '}') ++i; continue; }
        out.push_back(in[i]);
    }
    return out;
}

} // namespace

bool read(const std::string& path, std::vector<Segment>& out, std::string& err) {
    out.clear();
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "could not open SRT for reading: " + path; return false; }
    std::string content;
    { char buf[65536]; size_t n; while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) content.append(buf, n); }
    std::fclose(f);

    // Split into lines (handle CRLF/LF), then walk blocks.
    std::vector<std::string> lines;
    { std::string cur;
      for (char c : content) {
          if (c == '\n') { if (!cur.empty() && cur.back() == '\r') cur.pop_back(); lines.push_back(cur); cur.clear(); }
          else cur.push_back(c);
      }
      if (!cur.empty()) lines.push_back(cur);
    }

    for (size_t i = 0; i < lines.size(); ++i) {
        size_t arrow = lines[i].find("-->");
        if (arrow == std::string::npos) continue;
        double t0 = parse_ts(lines[i].substr(0, arrow));
        // second timestamp starts after the arrow (skip spaces)
        size_t j = arrow + 3; while (j < lines[i].size() && lines[i][j] == ' ') ++j;
        double t1 = parse_ts(lines[i].substr(j));
        if (t0 < 0 || t1 < 0) continue;
        Segment s; s.t0 = t0; s.t1 = t1;
        for (size_t k = i + 1; k < lines.size() && !lines[k].empty(); ++k) {
            if (!s.text.empty()) s.text.push_back(' ');
            s.text += strip_markup(lines[k]);
        }
        out.push_back(std::move(s));
    }
    return true;
}

bool write(const std::vector<Segment>& segs, const std::string& path,
           int max_line_length, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "could not open output for writing: " + path; return false; }
    int idx = 1;
    long dropped_dup = 0, dropped_empty = 0; // for the diagnostic log below
    Deduper dedup; // collapse whisper's repeated hallucination cues (see srt.h)
    for (const auto& seg : segs) {
        std::string text = wrap(seg.text, max_line_length); // also trims/collapses
        if (text.empty()) { ++dropped_empty; continue; }
        if (dedup.is_duplicate(text)) { ++dropped_dup; continue; } // repeated hallucination
        // SRT uses CRLF line endings; convert any '\n' from wrapping to "\r\n".
        std::string crlf;
        for (char c : text) { if (c == '\n') crlf += "\r\n"; else crlf.push_back(c); }
        std::fprintf(f, "%d\r\n%s --> %s\r\n%s\r\n\r\n",
                     idx++,
                     format_timestamp(seg.t0).c_str(),
                     format_timestamp(seg.t1).c_str(),
                     crlf.c_str());
    }
    std::fclose(f);
    logging::logf("INFO", "srt: wrote %d cues (dropped %ld duplicate, %ld empty of %zu)",
                  idx - 1, dropped_dup, dropped_empty, segs.size());
    if (dropped_dup > 10 && dropped_dup > (idx - 1))
        logging::logf("WARN",
            "srt: dropped more duplicate cues (%ld) than it kept (%d) - whisper was "
            "likely stuck in a repetition loop over part of the audio",
            dropped_dup, idx - 1);
    return true;
}

} // namespace srt
