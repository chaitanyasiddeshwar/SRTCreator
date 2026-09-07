#include "srt.h"
#include <cstdio>
#include <vector>

namespace srt {

namespace {

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

bool write(const std::vector<Segment>& segs, const std::string& path,
           int max_line_length, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "could not open output for writing: " + path; return false; }
    int idx = 1;
    for (const auto& seg : segs) {
        std::string text = wrap(seg.text, max_line_length); // also trims/collapses
        if (text.empty()) continue;
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
    return true;
}

} // namespace srt
