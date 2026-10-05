"""Stage 3: blocking. Units = (source, name_k, platform_k). Candidates between units of different
sources: (a) exact name_k+platform_k; (b) same platform_k & rapidfuzz token_set_ratio>=75 on folded names;
(c) platform-less units vs same name_k anywhere."""
import os, re, itertools, pandas as pd, numpy as np
from rapidfuzz import process, fuzz
W = os.path.dirname(os.path.abspath(__file__))
R = pd.read_pickle(f"{W}/state/norm.pkl")
R = R[~R.junk].copy()
R["unit"] = R.source + "|" + R.name_k.fillna("") + "|" + R.platform_k.fillna("")
U = R.groupby("unit").agg(source=("source","first"), name_k=("name_k","first"), platform_k=("platform_k","first"),
                          name=("name_c","first"), n=("rid","size")).reset_index()
U.to_pickle(f"{W}/state/units.pkl")
def norm_name(s): return re.sub(r'[^a-z0-9 ]', ' ', s.lower())
U["nn"] = U.name.map(norm_name)
pairs = {}
def add(a, b, how):
    if a == b: return
    k = tuple(sorted((a, b))); pairs.setdefault(k, set()).add(how)
# (a)+(b) per platform
for pk, g in U[U.platform_k.notna()].groupby("platform_k"):
    srcs = g.source.unique()
    for s1, s2 in itertools.combinations(sorted(srcs), 2):
        A = g[g.source == s1]; B = g[g.source == s2]
        M = process.cdist(A.nn.tolist(), B.nn.tolist(), scorer=fuzz.token_set_ratio, score_cutoff=75, workers=1)
        ii, jj = np.nonzero(M)
        for i, j in zip(ii, jj):
            add(A.unit.iat[i], B.unit.iat[j], "exact" if A.name_k.iat[i] == B.name_k.iat[j] else "fuzzy")
# (c) platform-less units
byname = U.groupby("name_k").unit.apply(list).to_dict()
for _, r in U[U.platform_k.isna()].iterrows():
    for u in byname.get(r.name_k, []):
        if not u.startswith(r.source + "|"): add(r.unit, u, "noplat")
C = pd.DataFrame([(a, b, "+".join(sorted(h))) for (a, b), h in pairs.items()], columns=["u1","u2","how"])
C.to_pickle(f"{W}/state/unit_candidates.pkl")
print(C.how.value_counts())
print("units", len(U), U.groupby("source").size().to_dict())
