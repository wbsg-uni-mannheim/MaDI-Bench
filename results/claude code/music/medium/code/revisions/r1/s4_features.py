"""Stage 4a: pairwise comparison features for every blocking candidate (interpretable, no training)."""
import pandas as pd, numpy as np, os, re
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

W = os.path.dirname(os.path.abspath(__file__))
SRCS = ["discogs", "lastfm", "musicbrainz"]
STOP_ART = {"the", "and", "feat", "featuring", "ft", "with", "vs", "presents"}
VARIOUS = {"various", "various artists", "va", "v a"}

def name_sim(a, b, drop_a, drop_b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    s = fuzz.token_sort_ratio(a, b) / 100
    ta, tb = a.split(), b.split()
    sa, sb = set(ta), set(tb)
    # one word dropped by noise (raw title shows a double space): subset with one missing token
    for small, big, drop in ((sa, sb, drop_a), (sb, sa, drop_b)):
        if small < big and len(big - small) == 1 and len(ta) == len(sa) and len(tb) == len(sb):
            # exactly one word missing: with the double-space drop marker it is very likely noise;
            # without it (word dropped at the start/end) it is weaker evidence
            s = max(s, 0.97 if drop else 0.9 if len(small) >= 2 else s)
    return s

def artist_tokens(k):
    return [t for t in k.split() if t not in STOP_ART]

def tok_match(x, y):
    if x == y: return True
    if len(x) == 1 or len(y) == 1:          # abbreviated initial 'F. McDonald'
        return x[0] == y[0]
    return len(x) >= 4 and len(y) >= 4 and fuzz.ratio(x, y) >= 80

def artist_sim(a, b):
    if not a or not b:
        return np.nan
    if a == b:
        return 1.0
    va, vb = a in VARIOUS, b in VARIOUS
    if va or vb:
        return 1.0 if va and vb else 0.0
    ta, tb = artist_tokens(a), artist_tokens(b)
    if not ta or not tb:
        return fuzz.ratio(a, b) / 100
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    used = set(); m = 0; full = 0; done = set()
    for strict in (True, False):          # full-token matches first, then initials / fuzzy
        for i, t in enumerate(small):
            if i in done: continue
            for j, u in enumerate(big):
                if j in used: continue
                if (t == u) if strict else tok_match(t, u):
                    used.add(j); done.add(i); m += 1; full += len(t) > 1 and len(u) > 1; break
    cont = m / len(small)
    if full == 0:            # only initials matched -> weak
        cont *= 0.5
    cover = m / len(big)
    return 0.7 * cont + 0.3 * cover

def track_sim(ta, tb):
    """fraction of tracks matched (fuzzy >= 85) over the larger list"""
    if not ta or not tb:
        return np.nan, np.nan
    A, B = list(ta), list(tb)
    small, big = (A, B) if len(A) <= len(B) else (B, A)
    bigset = set(big)
    m = 0
    for t in small:
        if t in bigset:
            m += 1
        else:
            r = process.extractOne(t, big, scorer=fuzz.ratio, score_cutoff=85)
            if r: m += 1
    return m / len(big), m / len(small)

def main():
    N = {s: pd.read_pickle(f"{W}/state/norm_{s}.pkl").set_index("id") for s in SRCS}
    C = pd.read_pickle(f"{W}/state/candidates.pkl")
    out = []
    for (sa, sb), g in C.groupby(["src1", "src2"]):
        A = N[sa].loc[g.id1].reset_index(); B = N[sb].loc[g.id2].reset_index()
        f = pd.DataFrame({"id1": g.id1.values, "id2": g.id2.values, "src1": sa, "src2": sb, "methods": g.methods.values})
        f["name_sim"] = [name_sim(a, b, da, db) for a, b, da, db in zip(A.name_key, B.name_key, A.name_dropmark, B.name_dropmark)]
        f["name_exact"] = (A.name_key.values == B.name_key.values)
        f["artist_sim"] = [artist_sim(a, b) for a, b in zip(A.artist_key, B.artist_key)]
        ts = [track_sim(a, b) for a, b in zip(A.track_keys, B.track_keys)]
        f["track_sim"] = [x[0] for x in ts]; f["track_cont"] = [x[1] for x in ts]
        f["ntr1"] = A.n_tracks.values; f["ntr2"] = B.n_tracks.values
        f["dur1"] = A.duration.values; f["dur2"] = B.duration.values
        f["date1"] = A.date.values; f["date2"] = B.date.values
        f["year1"] = A.year.values; f["year2"] = B.year.values
        f["month1"] = A.month.values; f["month2"] = B.month.values
        f["ctry1"] = A.country_key.values; f["ctry2"] = B.country_key.values
        out.append(f)
        print(sa, sb, len(f), flush=True)
    F = pd.concat(out, ignore_index=True)
    F.to_pickle(f"{W}/state/features.pkl")

if __name__ == "__main__":
    main()
