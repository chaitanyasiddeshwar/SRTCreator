#!/usr/bin/env python3
"""
Objective SRT timing/coverage comparison harness for SRTCreator.

Aligns a generated .srt against a ground-truth reference .srt using text
similarity constrained by time proximity, then reports the two failure modes
the user cares about:
  1. MISSING   - reference dialogue with no matching generated cue
  2. MISTIMED  - matched cues whose start time is off by a lot

Usage:
  python compare_srt.py <generated.srt> <reference.srt> [--json out.json] [--verbose]

Deterministic and self-contained (stdlib only) so the same inputs always give
the same score - suitable as a regression baseline.
"""
import re, sys, json, argparse
from difflib import SequenceMatcher

TS = re.compile(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)')

def parse_srt(path):
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        content = f.read()
    cues = []
    blocks = re.split(r'\r?\n\r?\n', content)
    for b in blocks:
        lines = [l for l in b.splitlines()]
        m = None; ti = -1
        for i, l in enumerate(lines):
            m = TS.search(l)
            if m: ti = i; break
        if not m: continue
        h0, m0, s0, ms0, h1, m1, s1, ms1 = map(int, m.groups())
        t0 = h0*3600 + m0*60 + s0 + ms0/1000.0
        t1 = h1*3600 + m1*60 + s1 + ms1/1000.0
        text = ' '.join(lines[ti+1:]).strip()
        if text:
            cues.append({'t0': t0, 't1': t1, 'text': text})
    return cues

def norm(s):
    s = re.sub(r'<[^>]+>', ' ', s)      # <i> tags
    s = re.sub(r'\{[^}]+\}', ' ', s)    # {\an8}
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def tokset(s):
    return set(norm(s).split())

def sim(a, b):
    """token F1 blended with sequence ratio - robust to word order & splits."""
    ta, tb = tokset(a), tokset(b)
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    inter = len(ta & tb)
    if inter == 0: return 0.0
    prec = inter/len(tb); rec = inter/len(ta)
    f1 = 2*prec*rec/(prec+rec)
    sq = SequenceMatcher(None, norm(a), norm(b)).ratio()
    return 0.6*f1 + 0.4*sq

def overlap(a, b):
    return min(a['t1'], b['t1']) - max(a['t0'], b['t0'])

def align(ref, gen, win=6.0, min_sim=0.34):
    """
    For each reference cue, find the best generated cue by text sim whose start
    is within `win` seconds OR which time-overlaps. Greedy, each gen used once.
    Returns list of (ref_idx, gen_idx|None, similarity).
    """
    used = [False]*len(gen)
    # index gen by time for a bounded search
    matches = []
    for ri, r in enumerate(ref):
        best_j, best_s = -1, 0.0
        for gj, g in enumerate(gen):
            if used[gj]: continue
            if g['t0'] > r['t0'] + 25 and overlap(r, g) <= 0: break  # gen sorted; past window
            near = abs(g['t0'] - r['t0']) <= win or overlap(r, g) > 0
            if not near: continue
            s = sim(r['text'], g['text'])
            # prefer higher sim, tie-break by time proximity
            if s > best_s + 1e-6 or (abs(s-best_s) < 1e-6 and best_j >= 0
                                     and abs(g['t0']-r['t0']) < abs(gen[best_j]['t0']-r['t0'])):
                best_s, best_j = s, gj
        if best_j >= 0 and best_s >= min_sim:
            used[best_j] = True
            matches.append((ri, best_j, best_s))
        else:
            matches.append((ri, None, best_s))
    return matches, used

def pct(x, n): return (100.0*x/n) if n else 0.0

# ---- Segmentation-independent word-level coverage ----
# Independent of how cues are chunked/merged. For each reference cue we gather
# the generated words that appear within a time window of it, and measure what
# fraction of the reference's words are present. Two windows decompose the two
# failure modes:
#   strict  (0.5s): word captured AT ~the right time  -> good
#   lenient (6.0s): word captured somewhere near      -> present but maybe off
#   truly_missing  = 1 - lenient     (dialogue not transcribed at all)
#   mistimed_only  = lenient - strict (transcribed but time is wrong)
def word_coverage(ref, gen, strict=0.5, lenient=6.0):
    gen_sorted = sorted(gen, key=lambda c: c['t0'])
    def gather(r, win):
        lo, hi = r['t0'] - win, r['t1'] + win
        bag = set()
        for g in gen_sorted:
            if g['t0'] > hi: break
            if g['t1'] < lo: continue
            bag |= tokset(g['text'])
        return bag
    tot = cs = cl = 0
    per = []
    for r in ref:
        rt = tokset(r['text'])
        if not rt: continue
        gs = gather(r, strict); gl = gather(r, lenient)
        s = sum(1 for w in rt if w in gs)
        l = sum(1 for w in rt if w in gl)
        tot += len(rt); cs += s; cl += l
        per.append((r, len(rt), s, l))
    return {
        'total_ref_words': tot,
        'strict_pct': round(pct(cs, tot), 2),
        'lenient_pct': round(pct(cl, tot), 2),
        'truly_missing_pct': round(pct(tot - cl, tot), 2),
        'mistimed_only_pct': round(pct(cl - cs, tot), 2),
    }, per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('generated'); ap.add_argument('reference')
    ap.add_argument('--json'); ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--win', type=float, default=6.0)
    a = ap.parse_args()

    gen = parse_srt(a.generated); ref = parse_srt(a.reference)
    gen.sort(key=lambda c: c['t0']); ref.sort(key=lambda c: c['t0'])

    matches, used = align(ref, gen, win=a.win)
    deltas = []       # signed start delta gen - ref, for matched
    matched_ref = 0
    missing = []      # unmatched reference cues
    for (ri, gj, s) in matches:
        if gj is None:
            missing.append(ref[ri]); continue
        matched_ref += 1
        deltas.append(gen[gj]['t0'] - ref[ri]['t0'])
    extra = [gen[j] for j in range(len(gen)) if not used[j]]  # unmatched gen cues

    ad = sorted(abs(d) for d in deltas)
    def med(v): return v[len(v)//2] if v else 0.0
    def p(v, q):
        if not v: return 0.0
        return v[min(len(v)-1, int(q*len(v)))]
    within = lambda thr: sum(1 for d in ad if d <= thr)

    n_ref, n_gen = len(ref), len(gen)
    summary = {
        'ref_cues': n_ref, 'gen_cues': n_gen,
        'matched': matched_ref, 'recall_pct': round(pct(matched_ref, n_ref), 2),
        'precision_pct': round(pct(matched_ref, n_gen), 2),
        'missing': len(missing),
        'extra_gen': len(extra),
        'timing': {
            'median_abs': round(med(ad), 3),
            'mean_abs': round(sum(ad)/len(ad), 3) if ad else 0.0,
            'median_signed': round(med(sorted(deltas)), 3),
            'p90_abs': round(p(ad, 0.90), 3),
            'within_0.5s_pct': round(pct(within(0.5), len(ad)), 2),
            'within_1.0s_pct': round(pct(within(1.0), len(ad)), 2),
            'over_2.0s': sum(1 for d in ad if d > 2.0),
        },
    }

    wc, wc_per = word_coverage(ref, gen)
    summary['word_coverage'] = wc

    print("="*64)
    print(f"  SRTCreator SRT comparison baseline")
    print("="*64)
    print( "  --- WORD-LEVEL COVERAGE (segmentation-independent) ---")
    print(f"    ref words captured  <=0.5s (well-timed): {wc['strict_pct']}%")
    print(f"    ref words captured  <=6.0s (present)   : {wc['lenient_pct']}%")
    print(f"    TRULY MISSING (not transcribed at all) : {wc['truly_missing_pct']}%")
    print(f"    MISTIMED ONLY (present but >0.5s off)  : {wc['mistimed_only_pct']}%")
    print("  "+"-"*60)
    print( "  --- CUE-LEVEL (sensitive to merging/splitting) ---")
    print(f"  reference cues : {n_ref}")
    print(f"  generated cues : {n_gen}")
    print(f"  matched        : {matched_ref}  (recall {summary['recall_pct']}% , precision {summary['precision_pct']}%)")
    print(f"  MISSING (ref w/ no gen) : {len(missing)}")
    print(f"  EXTRA   (gen w/ no ref) : {len(extra)}")
    print( "  --- timing of matched cues (|gen.start - ref.start|) ---")
    t = summary['timing']
    print(f"    median |d|   : {t['median_abs']}s     mean |d| : {t['mean_abs']}s")
    print(f"    median signed: {t['median_signed']}s  (+late / -early)")
    print(f"    p90 |d|      : {t['p90_abs']}s")
    print(f"    within 0.5s  : {t['within_0.5s_pct']}%   within 1.0s : {t['within_1.0s_pct']}%")
    print(f"    off > 2.0s   : {t['over_2.0s']} cues")
    print("="*64)

    # Composite regression score (higher = better). Built on the robust
    # word-level metrics so cue merging/splitting doesn't swing it: reward
    # well-timed captured words (strict), penalize truly-missing dialogue,
    # and factor cue-timing tightness.
    score = (0.60*wc['strict_pct']
             + 0.25*(100.0 - wc['truly_missing_pct'])
             + 0.15*t['within_1.0s_pct'])
    summary['composite_score'] = round(score, 2)
    print(f"  COMPOSITE SCORE: {round(score,2)}  (0.6*well-timed-words + 0.25*(100-missing) + 0.15*within1s)")
    print("="*64)

    if a.verbose:
        print("\n--- Worst mistimed matched cues (top 25 by |d|) ---")
        rows = []
        for (ri, gj, s) in matches:
            if gj is None: continue
            d = gen[gj]['t0'] - ref[ri]['t0']
            rows.append((abs(d), d, ref[ri], gen[gj], s))
        rows.sort(reverse=True)
        for absd, d, r, g, s in rows[:25]:
            print(f"  d{d:+6.2f}s sim{s:.2f} | ref {r['t0']:8.2f} \"{norm(r['text'])[:44]}\"")
            print(f"                | gen {g['t0']:8.2f} \"{norm(g['text'])[:44]}\"")
        print(f"\n--- Missing reference dialogue (first 40 of {len(missing)}) ---")
        for r in missing[:40]:
            print(f"  ref {r['t0']:8.2f}-{r['t1']:7.2f}  \"{norm(r['text'])[:56]}\"")

    if a.json:
        out = dict(summary)
        out['missing_list'] = [{'t0': r['t0'], 't1': r['t1'], 'text': r['text']} for r in missing]
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")

if __name__ == '__main__':
    main()
