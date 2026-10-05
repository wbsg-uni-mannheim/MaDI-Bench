"""Stage 5: pairwise scoring with hand-set, interpretable evidence weights (no training)."""
import pandas as pd, numpy as np, json, re, sys
F = pd.read_pickle("work/state/features.pkl")
R = pd.read_pickle("work/state/records.pkl").set_index("id")
NM = R.k_name.to_dict(); CT = R.country.to_dict()
MARKET = re.compile(r"Europe|&|,|America \(|Asia|Africa|Scandinavia|Benelux|Australasia|South America")
def extra_numbers(a, b):
    ta, tb = set(a.split()), set(b.split())
    return any(re.fullmatch(r"\d+|[ivx]+|vol|part|pt", t) for t in ta ^ tb)
lf = F.sp.str.contains("lastfm").values
SRC = R.source.to_dict()
def lf_contained(a, b):
    """lastfm titles lose words/get truncated: lastfm tokens must be contained in the other title
    (and cover at least half of it) for the subset relation to count as agreement."""
    if SRC[b] == "lastfm": a, b = b, a
    ta, tb = NM[a].split(), set(NM[b].split())
    return len(ta) > 0 and all(t in tb for t in ta) and len(set(ta)) >= 0.5 * len(tb)
cont = np.array([lf_contained(a, b) if l else False for a, b, l in zip(F.id1, F.id2, lf)])
name_s = np.where(lf, np.maximum(F.name_ratio, np.where(cont, 0.95, 0.0)), np.maximum(F.name_ratio, 0.9 * F.name_tset))
num = np.array([extra_numbers(NM[a], NM[b]) for a, b in zip(F.id1, F.id2)])
name_s = np.where(num, np.minimum(name_s, 0.7), name_s)
F["name_s"] = name_s
art = F.artist_sim.fillna(0.5).values
both_tr = (F.ntr1 > 0) & (F.ntr2 > 0)
trk = F.trk_max.fillna(0).values
t_term = np.where(both_tr, 2.0 * trk - np.where((trk < 0.3) & (F.ntr1 >= 3) & (F.ntr2 >= 3), 1.0, 0.0), 0.0)
dd, dr = F.dur_diff.values, F.dur_rel.values
d_term = np.where(np.isnan(dd), 0.0, np.where(dd <= 5, 1.0, np.where(dr <= 0.05, 0.5, np.where(dr > 0.2, -1.0, 0.0))))
yd = F.year_diff.values
y_term = np.where(np.isnan(yd), 0.0, np.where(yd == 0, 0.5, np.where(yd <= 1, 0.0, -1.5))) + np.nan_to_num(F.date_eq.values) * 0.5
ce = F.ctry_eq.values
mk = np.array([bool(MARKET.search(CT[a]) or MARKET.search(CT[b])) for a, b in zip(F.id1, F.id2)])
c_term = np.where(np.isnan(ce), 0.0, np.where(ce == 1, 0.5, np.where(mk, 0.0, -0.5)))
F["score"] = 2 * name_s + 2 * art + t_term + d_term + y_term + c_term
F["gate"] = (name_s >= 0.75) & (art >= 0.75)
F.to_pickle("work/state/scored.pkl")
G = F[F.gate]
print(G.groupby("sp").score.describe())
print(np.histogram(G.score, bins=np.arange(0, 10, 0.5)))
