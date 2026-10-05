"""Stage 4a: pair features for every blocking candidate."""
import pandas as pd, numpy as np, re
from rapidfuzz import fuzz, process
R = pd.read_pickle("work/state/records.pkl").set_index("id")
C = pd.read_csv("work/state/candidates.csv")

def akey(s):  # artist comparison tokens: drop discogs '(2)' disambiguators, 'the'
    s = re.sub(r"\b\d+\b$", "", s).strip()
    return [x for x in s.split() if x != "and"] or s.split()
AK = {i: akey(r) for i, r in R.k_artist.items()}
# discogs multi-artist credits are '|'-separated: also compare each credited artist separately
AKP = {i: [akey(" ".join(re.sub(r"[^a-z0-9]+", " ", p.lower()).split())) for p in a.split("|")] if "|" in a else []
       for i, a in R.artist.items()}
T = R.k_tracks.to_dict(); NM = R.k_name.to_dict(); SRC = R.source.to_dict()
DUR = R.duration.to_dict(); DATE = R.date.to_dict(); CTRY = R.country.to_dict()

def tok_match(a, b):
    return a == b or (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b)) \
        or (len(a) > 3 and len(b) > 3 and fuzz.ratio(a, b) >= 85)
def artist_sim(x, y):   # best of: as-is, and with a leading 'the'/'t' (lastfm 'T.' = 'The') dropped
    if not x or not y: return np.nan
    dt = lambda t: t[1:] if len(t) > 1 and t[0] in ("the", "t") else t
    return max(artist_sim0(x, y), artist_sim0(dt(x), dt(y)))
def artist_sim0(x, y):
    cx, cy = "".join(x), "".join(y)                        # 'c a t' (C/A/T) vs 'cat'
    if len(cx) >= 3 and cx == cy: return 1.0
    if not x or not y: return np.nan
    if "various" in x or "various" in y: return 1.0 if ("various" in x and "various" in y) else 0.0
    used = set(); m = 0
    for a in x:
        for j, b in enumerate(y):
            if j not in used and tok_match(a, b): used.add(j); m += 1; break
    return m / max(len(x), len(y))
def track_sim(x, y):
    if not x or not y: return np.nan, np.nan
    used = set(); m = 0
    for a in x:
        best = process.extractOne(a, [b if j not in used else "\x00" for j, b in enumerate(y)], scorer=fuzz.ratio)
        if best and best[1] >= 80: used.add(best[2]); m += 1
    return m / max(len(x), len(y)), m / min(len(x), len(y))
out = []
for a, b in zip(C.id1, C.id2):
    na, nb = NM[a], NM[b]
    ns = fuzz.ratio(na, nb) / 100; nts = fuzz.token_set_ratio(na, nb) / 100
    ars = artist_sim(AK[a], AK[b])
    for pa in AKP[a]: ars = max(ars, artist_sim(pa, AK[b]))
    for pb in AKP[b]: ars = max(ars, artist_sim(AK[a], pb))
    out.append((ns, nts, ars))
F = pd.DataFrame(out, columns=["name_ratio", "name_tset", "artist_sim"])
F = pd.concat([C, F], axis=1)
# expensive track sims only where name or artist evidence is non-trivial, or tracks view proposed it
need = (F.name_tset >= 0.6) | (F.artist_sim >= 0.5) | F.views.str.contains("tracks")
ts = [track_sim(T[a], T[b]) if n else (np.nan, np.nan) for a, b, n in zip(F.id1, F.id2, need)]
F["trk_max"] = [x[0] for x in ts]; F["trk_min"] = [x[1] for x in ts]
F["ntr1"] = F.id1.map(lambda i: len(T[i])); F["ntr2"] = F.id2.map(lambda i: len(T[i]))
d1 = F.id1.map(DUR); d2 = F.id2.map(DUR); F["dur_diff"] = (d1 - d2).abs()
F["dur_rel"] = F.dur_diff / np.maximum(d1, d2)
da = F.id1.map(DATE); db = F.id2.map(DATE)
has = (da != "") & (db != "")
F["year_eq"] = np.where(has, (da.str[:4] == db.str[:4]).astype(float), np.nan)
F["year_diff"] = np.where(has, (da.str[:4].replace("", "0").astype(int) - db.str[:4].replace("", "0").astype(int)).abs(), np.nan)
full = has & (da.str.len() == 10) & (db.str.len() == 10)
F["date_eq"] = np.where(full, (da == db).astype(float), np.nan)
ca = F.id1.map(CTRY); cb = F.id2.map(CTRY); hc = (ca != "") & (cb != "")
F["ctry_eq"] = np.where(hc, (ca == cb).astype(float), np.nan)
F["sp"] = F.id1.map(SRC) + "-" + F.id2.map(SRC)
F.to_pickle("work/state/features.pkl")
print(F.describe().T.to_string())
