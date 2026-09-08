#include "timeline.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace timeline {

namespace {

std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if ((unsigned char)c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
            out += buf;
        } else {
            out += c;
        }
    }
    return out;
}

float compute_avg_db(const std::vector<float>& pcm, double t0, double t1) {
    if (pcm.empty() || t0 >= t1) return -90.0f;
    size_t idx0 = (size_t)std::max(0.0, t0 * 16000.0);
    size_t idx1 = (size_t)std::min((double)pcm.size(), t1 * 16000.0);
    if (idx1 <= idx0) return -90.0f;
    double sum = 0.0;
    for (size_t i = idx0; i < idx1; ++i) {
        sum += (double)pcm[i] * pcm[i];
    }
    double rms = std::sqrt(sum / (idx1 - idx0));
    return (rms > 1e-9) ? (float)(20.0 * std::log10(rms)) : -90.0f;
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

std::string join_words(const std::vector<std::string>& w, size_t from, size_t to) {
    std::string s;
    for (size_t i = from; i < to && i < w.size(); ++i) {
        if (!s.empty()) s.push_back(' ');
        s += w[i];
    }
    return s;
}

} // namespace

TimelineMap build_timeline(const std::string& media_path,
                           const std::vector<float>& pcm,
                           const std::vector<transcribe::VadRegion>& raw_vad) {
    TimelineMap map;
    map.media_path = media_path;
    double total_sec = pcm.size() / 16000.0;
    map.analysis.total_duration = total_sec;

    if (total_sec <= 0.0) return map;

    // Filter and merge close VAD regions (gap < 0.15s)
    std::vector<transcribe::VadRegion> vad = raw_vad;
    std::sort(vad.begin(), vad.end(), [](const auto& a, const auto& b) { return a.t0 < b.t0; });

    std::vector<transcribe::VadRegion> merged;
    for (const auto& r : vad) {
        if (r.t1 <= r.t0) continue;
        if (!merged.empty() && r.t0 <= merged.back().t1 + 0.150) {
            merged.back().t1 = std::max(merged.back().t1, r.t1);
        } else {
            merged.push_back(r);
        }
    }

    // Partition [0, total_sec] into alternating Silence and Vocal
    double cur = 0.0;
    for (const auto& r : merged) {
        double v_start = std::max(cur, r.t0);
        double v_end   = std::min(total_sec, r.t1);

        if (v_start > cur + 0.030) {
            Interval sil;
            sil.type = IntervalType::Silence;
            sil.t0 = cur;
            sil.t1 = v_start;
            sil.avg_db = compute_avg_db(pcm, sil.t0, sil.t1);
            map.intervals.push_back(sil);
        }

        if (v_end > v_start + 0.020) {
            Interval voc;
            voc.type = IntervalType::Vocal;
            voc.t0 = v_start;
            voc.t1 = v_end;
            voc.avg_db = compute_avg_db(pcm, voc.t0, voc.t1);
            map.intervals.push_back(voc);
            cur = v_end;
        }
    }

    if (cur < total_sec - 0.030) {
        Interval sil;
        sil.type = IntervalType::Silence;
        sil.t0 = cur;
        sil.t1 = total_sec;
        sil.avg_db = compute_avg_db(pcm, sil.t0, sil.t1);
        map.intervals.push_back(sil);
    }

    // Compute aggregate analysis metrics
    for (const auto& iv : map.intervals) {
        if (iv.type == IntervalType::Vocal) {
            map.analysis.total_vocal_sec += iv.duration();
            map.analysis.vocal_count++;
        } else {
            map.analysis.total_silence_sec += iv.duration();
            map.analysis.silence_count++;
        }
    }
    if (total_sec > 0.0) {
        map.analysis.vocal_coverage_pct = (map.analysis.total_vocal_sec / total_sec) * 100.0;
    }

    return map;
}

bool sanitize_and_split(std::vector<srt::Segment>& segments,
                        TimelineMap& map,
                        double split_pause_threshold) {
    if (segments.empty() || map.intervals.empty()) return false;

    std::vector<srt::Segment> out;
    out.reserve(segments.size() + 16);

    for (size_t i = 0; i < segments.size(); ++i) {
        auto c = segments[i];
        if (c.t0 >= c.t1 || c.text.empty()) continue;

        // 1. Hallucination Check: compute total overlap with vocal regions
        double vocal_overlap = 0.0;
        for (const auto& iv : map.intervals) {
            if (iv.type != IntervalType::Vocal) continue;
            double ov = std::min(c.t1, iv.t1) - std::max(c.t0, iv.t0);
            if (ov > 0.0) vocal_overlap += ov;
        }

        // If cue has virtually no overlap with speech (< 0.05s) and duration >= 0.5s:
        if (vocal_overlap < 0.050 && c.t1 - c.t0 >= 0.50) {
            Action act;
            act.cue_index = (int)i + 1;
            act.action_type = "drop_hallucination";
            act.orig_t0 = c.t0; act.orig_t1 = c.t1;
            act.new_t0 = 0.0;  act.new_t1 = 0.0;
            act.text = c.text;
            act.reason = "100% in silence (no vocal energy)";
            map.actions.push_back(act);
            continue; // dropped
        }

        // 2. Leading Silence Snapping:
        // If c.t0 starts inside a silence interval that directly precedes speech:
        for (const auto& iv : map.intervals) {
            if (iv.type == IntervalType::Silence && c.t0 >= iv.t0 && c.t0 < iv.t1) {
                // If silence ends before cue ends:
                if (iv.t1 < c.t1 && (iv.t1 - c.t0) > 0.150) {
                    double snapped = std::max(c.t0, iv.t1 - 0.050);
                    if (snapped < c.t1 - 0.100) {
                        Action act;
                        act.cue_index = (int)i + 1;
                        act.action_type = "snap_leading";
                        act.orig_t0 = c.t0; act.orig_t1 = c.t1;
                        c.t0 = snapped;
                        act.new_t0 = c.t0; act.new_t1 = c.t1;
                        act.text = c.text;
                        act.reason = "snapped start forward to vocal onset";
                        map.actions.push_back(act);
                    }
                }
                break;
            }
        }

        // 3. Option 2: Mid-sentence Pause Splitting
        // Check if there is an internal silence gap >= split_pause_threshold inside this cue
        bool was_split = false;
        for (const auto& iv : map.intervals) {
            if (iv.type == IntervalType::Silence && iv.duration() >= split_pause_threshold) {
                // Check if silence is strictly inside cue (at least 0.25s speech on either side)
                if (c.t0 < iv.t0 - 0.250 && c.t1 > iv.t1 + 0.250) {
                    auto words = split_words(c.text);
                    if (words.size() >= 2) {
                        double dur1 = iv.t0 - c.t0;
                        double dur2 = c.t1 - iv.t1;
                        double ratio = dur1 / (dur1 + dur2);
                        size_t split_idx = (size_t)std::round(ratio * words.size());
                        split_idx = std::clamp(split_idx, (size_t)1, words.size() - 1);

                        // Look for punctuation nearby to make split natural
                        for (size_t k = 1; k < words.size(); ++k) {
                            char last_ch = words[k - 1].back();
                            if (last_ch == '.' || last_ch == ',' || last_ch == ';' || last_ch == '!' || last_ch == '?') {
                                if (std::abs((int)k - (int)split_idx) <= 2) {
                                    split_idx = k;
                                    break;
                                }
                            }
                        }

                        srt::Segment c1;
                        c1.t0 = c.t0;
                        c1.t1 = std::min(c.t1, iv.t0 + 0.150);
                        c1.text = join_words(words, 0, split_idx);

                        srt::Segment c2;
                        c2.t0 = std::max(c1.t1 + 0.050, iv.t1 - 0.050);
                        c2.t1 = c.t1;
                        c2.text = join_words(words, split_idx, words.size());

                        Action act;
                        act.cue_index = (int)i + 1;
                        act.action_type = "split_pause";
                        act.orig_t0 = c.t0; act.orig_t1 = c.t1;
                        act.new_t0 = c1.t0; act.new_t1 = c2.t1;
                        act.text = c.text;
                        act.reason = "split cue across " + std::to_string(iv.duration()).substr(0, 4) + "s silence pause";
                        map.actions.push_back(act);

                        out.push_back(c1);
                        out.push_back(c2);
                        was_split = true;
                        break;
                    }
                }
            }
        }
        if (was_split) continue;

        // 4. Trailing Silence Clamping:
        // Find the last vocal interval that overlaps with cue
        double last_voc_end = -1.0;
        for (const auto& iv : map.intervals) {
            if (iv.type != IntervalType::Vocal) continue;
            if (iv.t0 < c.t1 && iv.t1 > c.t0) {
                last_voc_end = std::max(last_voc_end, iv.t1);
            }
        }

        if (last_voc_end > c.t0 && c.t1 > last_voc_end + 0.200) {
            // Find silence gap immediately following last_voc_end
            double next_sil_end = c.t1;
            for (const auto& iv : map.intervals) {
                if (iv.type == IntervalType::Silence && std::abs(iv.t0 - last_voc_end) < 0.050) {
                    next_sil_end = iv.t1;
                    break;
                }
            }

            double min_read = std::min(1.40, std::max(0.80, 0.04 * c.text.size() + 0.40));
            double target_end = std::max(c.t0 + min_read, last_voc_end + 0.150);
            target_end = std::min(target_end, next_sil_end - 0.050);

            if (target_end < c.t1) {
                Action act;
                act.cue_index = (int)i + 1;
                act.action_type = "clamp_trailing";
                act.orig_t0 = c.t0; act.orig_t1 = c.t1;
                c.t1 = target_end;
                act.new_t0 = c.t0; act.new_t1 = c.t1;
                act.text = c.text;
                act.reason = "clamped trailing silence";
                map.actions.push_back(act);
            }
        }

        out.push_back(c);
    }

    // 5. Final Reading Duration & Inter-Cue Gap Enforcer
    for (size_t k = 0; k < out.size(); ++k) {
        if (out[k].t0 >= out[k].t1) continue;
        double min_read = std::min(2.5, std::max(0.80, 0.04 * (double)out[k].text.size() + 0.50));
        double target_end = std::max(out[k].t1, out[k].t0 + min_read);
        if (k + 1 < out.size() && out[k + 1].t0 > out[k].t0) {
            target_end = std::min(target_end, out[k + 1].t0 - 0.050);
        }
        if (target_end > out[k].t0 + 0.200) {
            out[k].t1 = target_end;
        }
    }

    segments = std::move(out);
    return true;
}

void find_missing_vocal_regions(const std::vector<srt::Segment>& segments,
                                TimelineMap& map,
                                double min_duration,
                                double min_coverage) {
    map.missing_vocal.clear();

    for (const auto& iv : map.intervals) {
        if (iv.type != IntervalType::Vocal) continue;
        double dur = iv.duration();
        if (dur < min_duration) continue;

        // Calculate total subtitle coverage within this vocal interval
        double covered = 0.0;
        for (const auto& s : segments) {
            double ov = std::min(s.t1, iv.t1) - std::max(s.t0, iv.t0);
            if (ov > 0.0) covered += ov;
        }

        double ratio = covered / dur;
        if (ratio < min_coverage) {
            MissingRegion mr;
            mr.t0 = iv.t0;
            mr.t1 = iv.t1;
            mr.avg_db = iv.avg_db;
            mr.coverage = ratio;
            map.missing_vocal.push_back(mr);
        }
    }
}

bool infill_missing_regions(const std::vector<float>& pcm,
                            transcribe::Session& session,
                            const transcribe::Options& base_opts,
                            TimelineMap& map,
                            std::vector<srt::Segment>& segments,
                            std::function<void(const std::string&)> on_log) {
    if (map.missing_vocal.empty() || pcm.empty()) return true;

    int recovered_cues = 0;
    for (size_t idx = 0; idx < map.missing_vocal.size(); ++idx) {
        const auto& mr = map.missing_vocal[idx];
        if (mr.duration() < 0.6 || mr.duration() > 30.0) continue;

        double pad = 0.350;
        double t_start = std::max(0.0, mr.t0 - pad);
        double t_end   = std::min((double)pcm.size() / 16000.0, mr.t1 + pad);

        size_t sample_start = (size_t)(t_start * 16000.0);
        size_t sample_end   = (size_t)(t_end * 16000.0);
        if (sample_end <= sample_start) continue;

        std::vector<float> slice(pcm.begin() + sample_start, pcm.begin() + sample_end);

        transcribe::Options opt = base_opts;
        opt.vad = false; // already known speech
        opt.vad_model_path.clear();
        opt.vad_regions = nullptr;
        opt.on_progress = nullptr;
        opt.on_segment  = nullptr;
        opt.word_timestamps = false;

        std::vector<srt::Segment> chunk_segs;
        std::string err;
        if (session.transcribe(slice, opt, chunk_segs, err)) {
            for (auto& s : chunk_segs) {
                if (s.text.empty()) continue;
                s.t0 += t_start;
                s.t1 += t_start;

                // Check for duplicate or heavy overlap with existing cues
                bool duplicate = false;
                for (const auto& ex : segments) {
                    double ov = std::min(s.t1, ex.t1) - std::max(s.t0, ex.t0);
                    if (ov > 0.40 * (s.t1 - s.t0)) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    Action act;
                    act.cue_index = (int)segments.size() + 1;
                    act.action_type = "infill_recovered";
                    act.orig_t0 = mr.t0; act.orig_t1 = mr.t1;
                    act.new_t0 = s.t0;   act.new_t1 = s.t1;
                    act.text = s.text;
                    act.reason = "recovered missing vocal region";
                    map.actions.push_back(act);

                    segments.push_back(s);
                    recovered_cues++;
                    if (on_log) {
                        on_log("Infill recovered dialogue at [" + srt::format_timestamp(s.t0).substr(0, 8) + "]: " + s.text);
                    }
                }
            }
        }
        std::vector<float>().swap(slice); // release chunk slice buffer immediately
    }

    if (recovered_cues > 0) {
        std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) { return a.t0 < b.t0; });
        logging::logf("INFO", "Pass 3: infilled %d recovered dialogue cues", recovered_cues);
    }
    return true;
}

bool infill_missing_regions(const std::vector<float>& pcm,
                            const transcribe::Options& base_opts,
                            TimelineMap& map,
                            std::vector<srt::Segment>& segments,
                            std::function<void(const std::string&)> on_log) {
    transcribe::Session session;
    std::string err;
    if (!session.init(base_opts, err)) {
        logging::error("infill: failed to init whisper session: " + err);
        return false;
    }
    return infill_missing_regions(pcm, session, base_opts, map, segments, on_log);
}

bool write_json(const TimelineMap& map, const std::string& json_path, std::string& err) {
    FILE* f = std::fopen(json_path.c_str(), "w");
    if (!f) {
        err = "could not open json file for writing: " + json_path;
        return false;
    }

    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"media\": \"%s\",\n", escape_json(map.media_path).c_str());
    std::fprintf(f, "  \"analysis\": {\n");
    std::fprintf(f, "    \"total_duration\": %.3f,\n", map.analysis.total_duration);
    std::fprintf(f, "    \"total_vocal_sec\": %.3f,\n", map.analysis.total_vocal_sec);
    std::fprintf(f, "    \"total_silence_sec\": %.3f,\n", map.analysis.total_silence_sec);
    std::fprintf(f, "    \"vocal_coverage_pct\": %.1f,\n", map.analysis.vocal_coverage_pct);
    std::fprintf(f, "    \"vocal_region_count\": %d,\n", map.analysis.vocal_count);
    std::fprintf(f, "    \"silence_region_count\": %d,\n", map.analysis.silence_count);
    std::fprintf(f, "    \"actions_count\": %zu,\n", map.actions.size());
    std::fprintf(f, "    \"missing_vocal_count\": %zu\n", map.missing_vocal.size());
    std::fprintf(f, "  },\n");

    // Timeline array
    std::fprintf(f, "  \"timeline\": [\n");
    for (size_t i = 0; i < map.intervals.size(); ++i) {
        const auto& iv = map.intervals[i];
        std::fprintf(f, "    {\"type\": \"%s\", \"start\": %.3f, \"end\": %.3f, \"duration\": %.3f, \"avg_db\": %.1f}%s\n",
                     iv.type == IntervalType::Vocal ? "vocal" : "silence",
                     iv.t0, iv.t1, iv.duration(), iv.avg_db,
                     (i + 1 < map.intervals.size()) ? "," : "");
    }
    std::fprintf(f, "  ],\n");

    // Missing vocal regions array
    std::fprintf(f, "  \"missing_vocal_regions\": [\n");
    for (size_t i = 0; i < map.missing_vocal.size(); ++i) {
        const auto& mr = map.missing_vocal[i];
        std::fprintf(f, "    {\"start\": %.3f, \"end\": %.3f, \"duration\": %.3f, \"avg_db\": %.1f, \"coverage\": %.2f}%s\n",
                     mr.t0, mr.t1, mr.duration(), mr.avg_db, mr.coverage,
                     (i + 1 < map.missing_vocal.size()) ? "," : "");
    }
    std::fprintf(f, "  ],\n");

    // Actions array
    std::fprintf(f, "  \"actions\": [\n");
    for (size_t i = 0; i < map.actions.size(); ++i) {
        const auto& act = map.actions[i];
        std::fprintf(f, "    {\"cue\": %d, \"action\": \"%s\", \"orig\": [%.3f, %.3f], \"new\": [%.3f, %.3f], \"text\": \"%s\", \"reason\": \"%s\"}%s\n",
                     act.cue_index, act.action_type.c_str(),
                     act.orig_t0, act.orig_t1, act.new_t0, act.new_t1,
                     escape_json(act.text).c_str(), escape_json(act.reason).c_str(),
                     (i + 1 < map.actions.size()) ? "," : "");
    }
    std::fprintf(f, "  ]\n");
    std::fprintf(f, "}\n");

    std::fclose(f);
    logging::logf("INFO", "timeline: wrote JSON timeline map (%zu intervals, %zu actions, %zu missing vocal) to %s",
                  map.intervals.size(), map.actions.size(), map.missing_vocal.size(), json_path.c_str());
    return true;
}

ActionCounts count_actions(const TimelineMap& map) {
    ActionCounts ac;
    for (const auto& a : map.actions) {
        if (a.action_type == "clamp_trailing") ac.clamped++;
        else if (a.action_type == "snap_leading") ac.snapped++;
        else if (a.action_type == "split_pause") ac.split++;
        else if (a.action_type == "drop_hallucination") ac.dropped++;
        else if (a.action_type == "infill_recovered") ac.infilled++;
    }
    return ac;
}

} // namespace timeline
