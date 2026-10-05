"""Stage 4a: pairwise evidence for every blocking candidate -> work/state/s4_features.pkl"""
import os, re, time
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = f"{BASE}/work/state"
GENERIC_ARTISTS = {"various", "various artists", "va", "unknown artist", ""}


def artist_sim(a, b):
    """a, b: comparison keys (space separated tokens). Handles 'L. Garnier' ~ 'laurent garnier',
    word order swaps ('65 buck'), and OCR noise. Returns 0..1, or nan if either unknown."""
    if not a or not b:
        return np.nan
    if a in GENERIC_ARTISTS or b in GENERIC_ARTISTS:
        return 1.0 if a == b else np.nan
    base = max(fuzz.token_sort_ratio(a, b), fuzz.ratio(a, b)) / 100
    ta, tb = a.split(), b.split()
    if len(ta) > len(tb):
        ta, tb = tb, ta
    # token alignment allowing initials and fuzzy tokens
    used = set(); hit = 0.0
    for t in ta:
        best, bj = 0.0, None
        for j, u in enumerate(tb):
            if j in used:
                continue
            if t == u:
                s = 1.0
            elif len(t) == 1 and u.startswith(t):
                s = 0.9
            elif len(u) == 1 and t.startswith(u):
                s = 0.9
            else:
                s = fuzz.ratio(t, u) / 100
                s = s if s >= 0.75 else 0
            if s > best:
                best, bj = s, j
        if bj is not None:
            used.add(bj); hit += best
    # initials may only support a full-token match (single 't' must not explain 'tofubeats')
    full = any(len(t) > 1 and any(fuzz.ratio(t, u) >= 75 for u in tb if len(u) > 1) for t in ta)
    align = hit / max(len(ta), 1) if full else 0.0
    # penalise when the longer name has many unexplained tokens (except 'the', 'and')
    extra = [u for j, u in enumerate(tb) if j not in used and u not in ("the", "and", "a", "feat", "ft", "dj")]
    align *= 1.0 if not extra else (0.85 if len(ta) >= 1 and len(extra) == 1 else 0.6)
    return max(base, align)


OCR = str.maketrans({"0": "o", "1": "l", "5": "s", "3": "e", "8": "b", "4": "a", "7": "t", "6": "g"})


def ocr_norm(s):
    """digit->letter OCR confusions and e/c, a/o skeleton; used only as a second comparison view"""
    return s.translate(OCR).replace("c", "e").replace("o", "a") if s else s


def title_sim(a, b):
    if not a or not b:
        return np.nan
    return max(fuzz.token_set_ratio(a, b), fuzz.ratio(a, b)) / 100


def title_sim_strict(a, b):
    if not a or not b:
        return np.nan
    return max(fuzz.token_sort_ratio(a, b), fuzz.ratio(a, b)) / 100


def track_sim(la, lb, ba, bb):
    """la/lb: track keys, ba/bb: same with bracketed version info removed.
    Full-key fuzzy matches (ratio >= 80) count 1; matches only after removing '(... mix)' etc. count 0.5.
    -> (weighted matches, frac_of_shorter, frac_of_longer); nan when either list empty"""
    if not la or not lb:
        return np.nan, np.nan, np.nan
    M = process.cdist(la, lb, scorer=fuzz.ratio)
    B = process.cdist(ba, bb, scorer=fuzz.ratio)
    W = np.where(M >= 80, 1.0, np.where(B >= 90, 0.5, 0.0))
    pairs = sorted(((W[i, j], M[i, j], i, j) for i, j in zip(*np.nonzero(W))), reverse=True)
    ui, uj = set(), set()
    matched = 0.0
    for w, _, i, j in pairs:
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); matched += w
    return matched, matched / min(len(la), len(lb)), matched / max(len(la), len(lb))


def main():
    t0 = time.time()
    df = pd.read_pickle(f"{ST}/s2_all.pkl")
    c = pd.read_pickle(f"{ST}/s3_candidates.pkl")
    kt = df.k_title.fillna("").values
    ka = df.k_artist.fillna("").values
    ktr = df.k_tracks.values
    okt = [ocr_norm(t) for t in kt]
    # purely numeric title tokens (years, volume numbers) distinguish releases: 'Best Of 1974-1979' vs '1969-1974'
    nums = [frozenset(x for x in t.split() if x.isdigit()) for t in kt]
    ktr_set, ktr_base = [], []
    for l, lb in zip(ktr, df.k_tracks_base.values):
        seen, a1, b1 = set(), [], []
        for t, u in zip(l, lb):
            if t and t not in seen:
                seen.add(t); a1.append(t); b1.append(u)
        ktr_set.append(a1); ktr_base.append(b1)
    dur = df.duration_s.values
    yr = df.year.values
    dt = df.date.values
    prec = df.date_prec.values
    ctry = df.country.values
    # lastfm title prefix artist (e.g. 'John B -  ...') is extra artist evidence
    tpa = df.title_prefix_artist.map(lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip() if isinstance(x, str) else "").values
    f = {k: [] for k in ["t_sim", "t_strict", "a_sim", "tr_n", "tr_short", "tr_long", "n_tr1", "n_tr2", "dur_diff", "dur_rel",
                         "year_diff", "date_eq", "country_eq", "t_in_tracks", "t_prefix", "num_conflict"]}
    for n, (i, j) in enumerate(zip(c.i1.values, c.i2.values)):
        f["t_sim"].append(max(title_sim(kt[i], kt[j]), title_sim(okt[i], okt[j])))
        f["t_strict"].append(max(title_sim_strict(kt[i], kt[j]), title_sim_strict(okt[i], okt[j])))
        ai, aj = ka[i] or tpa[i], ka[j] or tpa[j]
        s_a = artist_sim(ai, aj)
        if s_a == s_a and s_a < 0.9:
            s_a = max(s_a, artist_sim(ocr_norm(ai), ocr_norm(aj)))
        f["a_sim"].append(s_a)
        m, s, l = track_sim(ktr_set[i], ktr_set[j], ktr_base[i], ktr_base[j])
        f["tr_n"].append(m); f["tr_short"].append(s); f["tr_long"].append(l)
        f["n_tr1"].append(len(ktr_set[i])); f["n_tr2"].append(len(ktr_set[j]))
        if dur[i] == dur[i] and dur[j] == dur[j] and dur[i] and dur[j]:
            d = abs(dur[i] - dur[j]); f["dur_diff"].append(d); f["dur_rel"].append(d / max(dur[i], dur[j]))
        else:
            f["dur_diff"].append(np.nan); f["dur_rel"].append(np.nan)
        f["year_diff"].append(abs(yr[i] - yr[j]) if yr[i] == yr[i] and yr[j] == yr[j] and yr[i] and yr[j] else np.nan)
        if dt[i] and dt[j] and prec[i] == "day" and prec[j] == "day":
            f["date_eq"].append(float(dt[i] == dt[j]))
        else:
            f["date_eq"].append(np.nan)
        f["country_eq"].append(float(ctry[i] == ctry[j]) if ctry[i] and ctry[j] else np.nan)
        # lastfm sometimes names a release after one of its tracks ('Tempest' for 'Black Tempest')
        tt = max((fuzz.ratio(kt[i], x) for x in ktr_set[j]), default=0) if kt[i] else 0
        tt = max(tt, max((fuzz.ratio(kt[j], x) for x in ktr_set[i]), default=0) if kt[j] else 0)
        f["t_in_tracks"].append(tt / 100)
        a_, b_ = (kt[i], kt[j]) if len(kt[i]) <= len(kt[j]) else (kt[j], kt[i])
        ni, nj = nums[i], nums[j]
        f["num_conflict"].append(float(bool(ni) and bool(nj) and not (ni <= nj or nj <= ni)))
        f["t_prefix"].append(float(bool(a_) and (b_ + " ").startswith(a_ + " ")))
        if n % 100000 == 0:
            print(n, f"{time.time()-t0:.0f}s", flush=True)
    for k, v in f.items():
        c[k] = np.array(v, dtype=float)
    c.to_pickle(f"{ST}/s4_features.pkl")
    print(c.describe().T.to_string())


if __name__ == "__main__":
    main()
