#include "debug.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace debugreport {

namespace {

constexpr int    kRate    = 16000;
constexpr double kBinSec  = 0.020;          // 20 ms energy bins
constexpr double kThrDb   = -40.0;          // voiced threshold
const int        kBinN    = (int)(kRate * kBinSec); // samples per bin

// Per-bin RMS in dBFS (-inf floored to -90).
std::vector<double> energy_bins(const std::vector<float>& pcm) {
    std::vector<double> db;
    db.reserve(pcm.size() / kBinN + 1);
    for (size_t i = 0; i < pcm.size(); i += kBinN) {
        size_t end = std::min(pcm.size(), i + kBinN);
        double sum = 0.0;
        for (size_t j = i; j < end; ++j) sum += (double)pcm[j] * pcm[j];
        double rms = std::sqrt(sum / std::max<size_t>(1, end - i));
        db.push_back(rms > 1e-9 ? 20.0 * std::log10(rms) : -90.0);
    }
    return db;
}

double db_at(const std::vector<double>& db, double t) {
    int b = (int)(t / kBinSec);
    if (b < 0 || b >= (int)db.size()) return -90.0;
    return db[b];
}

// First time in [from, to] whose bin is >= threshold; -1 if none voiced.
double voiced_onset(const std::vector<double>& db, double from, double to) {
    if (from < 0) from = 0;
    int b0 = (int)(from / kBinSec), b1 = (int)(to / kBinSec);
    for (int b = b0; b <= b1 && b < (int)db.size(); ++b)
        if (b >= 0 && db[b] >= kThrDb) return b * kBinSec;
    return -1.0;
}

// Index of the VAD region covering t, or -1.
int vad_index(const std::vector<transcribe::VadRegion>& v, double t) {
    for (int i = 0; i < (int)v.size(); ++i)
        if (t >= v[i].t0 && t <= v[i].t1) return i;
    return -1;
}

// Nearest reference cue by start time; returns index or -1. Prefers overlap.
int nearest_ref(const std::vector<srt::Segment>& ref, const srt::Segment& c) {
    int best = -1; double bestd = 1e18;
    for (int i = 0; i < (int)ref.size(); ++i) {
        double ov = std::min(c.t1, ref[i].t1) - std::max(c.t0, ref[i].t0);
        double d  = ov > 0 ? 0.0 : std::abs(ref[i].t0 - c.t0); // overlap wins
        double key = d + (ov > 0 ? 0.0 : 0.0) + std::abs(ref[i].t0 - c.t0) * 1e-6;
        if (key < bestd) { bestd = key; best = i; }
    }
    return best;
}

// One-line energy strip for [t0, t1] at 100 ms/char, with markers.
std::string strip(const std::vector<double>& db, double t0, double t1,
                  double cue_start, const std::vector<transcribe::VadRegion>& vad) {
    std::string s;
    for (double t = t0; t < t1; t += 0.1) {
        // markers take priority over the level glyph
        bool at_cue = (t <= cue_start && cue_start < t + 0.1);
        double d = db_at(db, t);
        char g = d < -50 ? ' ' : d < -40 ? '.' : d < -30 ? ':' : d < -20 ? 'o' : '#';
        if (at_cue) g = '|';
        s.push_back(g);
    }
    (void)vad;
    return s;
}

std::string first_line(const std::string& text) {
    std::string t;
    for (char c : text) { if (c == '\n' || c == '\r') break; t.push_back(c); }
    if (t.size() > 60) t = t.substr(0, 57) + "...";
    return t;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

} // namespace

bool write(const std::string& path,
           const std::vector<float>& pcm16k,
           const std::vector<srt::Segment>& cues,
           const std::vector<transcribe::VadRegion>& vad,
           const std::vector<srt::Segment>* reference,
           const Info& info, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { err = "could not open debug report for writing: " + path; return false; }

    std::vector<double> db = energy_bins(pcm16k);
    double audio_sec = pcm16k.size() / (double)kRate;
    size_t voiced_bins = 0;
    for (double d : db) if (d >= kThrDb) ++voiced_bins;

    std::fprintf(f, "SRTCreator debug report\n");
    std::fprintf(f, "model=%s isolate=%d center=%d vad=%d word_ts=%d\n",
                 info.model.c_str(), info.isolate, info.center, info.vad, info.word_ts);
    std::fprintf(f, "audio: %.1fs @%dHz mono | energy bin=%.0fms voiced_thr=%.0fdB\n",
                 audio_sec, kRate, kBinSec * 1000, kThrDb);
    std::fprintf(f, "voiced coverage: %.1f%% of audio\n\n",
                 db.empty() ? 0.0 : 100.0 * voiced_bins / db.size());

    // --- VAD regions ---
    if (!vad.empty()) {
        std::fprintf(f, "== VAD speech regions: %zu ==\n", vad.size());
        std::fprintf(f, "  %-5s %9s %9s %7s\n", "#", "start", "end", "dur");
        for (int i = 0; i < (int)vad.size(); ++i)
            std::fprintf(f, "  %-5d %9.2f %9.2f %7.2f\n", i, vad[i].t0, vad[i].t1, vad[i].t1 - vad[i].t0);
        std::fprintf(f, "\n");
    } else {
        std::fprintf(f, "== VAD: not used (whisper saw the whole stream) ==\n\n");
    }

    // --- Cue table ---
    bool have_ref = reference && !reference->empty();
    std::fprintf(f, "== Cues (%zu): timing vs audio energy%s ==\n", cues.size(),
                 have_ref ? " vs reference" : "");
    std::fprintf(f, "  %-4s %9s %9s %9s %8s %5s", "idx", "start", "end", "early_by", "e@start", "vadR");
    if (have_ref) std::fprintf(f, " %10s %8s", "ref_start", "d_ref");
    std::fprintf(f, "  text\n");

    std::vector<double> early_all, dref_all; int matched = 0;
    struct Flag { int idx; double start, metric; std::string text; };
    std::vector<Flag> flagged;

    for (int i = 0; i < (int)cues.size(); ++i) {
        const auto& c = cues[i];
        double onset = voiced_onset(db, c.t0 - 2.0, c.t1);
        double early = (onset >= 0) ? onset - c.t0 : 0.0; // + => voice starts after cue (early)
        int vr = vad_index(vad, c.t0);
        early_all.push_back(early);

        std::string vlbl = vr >= 0 ? ("R" + std::to_string(vr)) : "-";
        std::fprintf(f, "  %-4d %9.2f %9.2f %+9.2f %7.1fd %5s", i + 1, c.t0, c.t1, early,
                     db_at(db, c.t0), vlbl.c_str());
        double dref = 0.0; bool has_dref = false;
        if (have_ref) {
            int r = nearest_ref(*reference, c);
            if (r >= 0) {
                dref = c.t0 - (*reference)[r].t0; has_dref = true;
                dref_all.push_back(dref); ++matched;
                std::fprintf(f, " %10.2f %+8.2f", (*reference)[r].t0, dref);
            } else {
                std::fprintf(f, " %10s %8s", "-", "-");
            }
        }
        std::fprintf(f, "  %s\n", first_line(c.text).c_str());

        // Flag by the reliable signal: vs reference a cue >0.5s EARLY (d_ref < -0.5);
        // without a reference, fall back to the (residual-fooled) energy metric.
        if (has_dref) { if (dref < -0.5) flagged.push_back({ i + 1, c.t0, dref, first_line(c.text) }); }
        else if (early > 0.5)             flagged.push_back({ i + 1, c.t0, early, first_line(c.text) });
    }
    std::fprintf(f, "\n  early_by = voiced_onset(near cue) - cue_start; + => cue appears before voice\n");
    std::fprintf(f, "  e@start  = energy at cue_start (dB); low = silence there\n");
    std::fprintf(f, "  vadR     = VAD region covering cue_start ('-' none)\n");
    if (have_ref) std::fprintf(f, "  d_ref    = cue_start - nearest reference cue start (+ later, - earlier)\n");
    std::fprintf(f, "\n");

    // --- Flagged early cues, with an energy strip for context ---
    if (!flagged.empty()) {
        std::fprintf(f, "== Flagged: %zu cues appear >0.5s early (%s) ==\n", flagged.size(),
                     have_ref ? "vs reference" : "vs voiced-energy onset");
        std::fprintf(f, "  strip = energy cue_start-3s .. +2s @100ms; ' '<-50 '.'<-40 ':'<-30 'o'<-20 '#'>=-20 dB; '|'=cue_start\n");
        std::fprintf(f, "  (in isolation mode a run of '.'/':'/'o' before '|' is separation residual VAD mistook for speech)\n");
        for (const auto& fl : flagged) {
            std::fprintf(f, "  #%-4d start=%7.2f %s=%+.2f  %s\n", fl.idx, fl.start,
                         have_ref ? "d_ref" : "early", fl.metric, fl.text.c_str());
            std::fprintf(f, "        [%s]\n", strip(db, fl.start - 3.0, fl.start + 2.0, fl.start, vad).c_str());
        }
        std::fprintf(f, "\n");
    }

    // --- Summary ---
    std::fprintf(f, "== Summary ==\n");
    int early_ct = 0; for (double e : early_all) if (e > 0.5) ++early_ct;
    std::fprintf(f, "cues: %zu | median early_by: %+.2fs | early>0.5s: %d (%.1f%%)\n",
                 cues.size(), median(early_all), early_ct,
                 cues.empty() ? 0.0 : 100.0 * early_ct / cues.size());
    if (have_ref) {
        std::vector<double> absd; for (double d : dref_all) absd.push_back(std::abs(d));
        std::fprintf(f, "reference: matched %d/%zu | median |d_ref|: %.2fs | median d_ref: %+.2fs\n",
                     matched, cues.size(), median(absd), median(dref_all));
    }

    std::fclose(f);
    return true;
}

} // namespace debugreport
