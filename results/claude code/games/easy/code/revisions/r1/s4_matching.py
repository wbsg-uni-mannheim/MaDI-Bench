"""Stage 4: unit-level matching decisions (deterministic rules, no training).
exact  : same name key + same platform -> accept unless remake contradiction
         (both years known, |dy|>=5 and critic scores both known and differ by >3; dbpedia years ignored
          because dbpedia rows are cross-products of release dates).
fuzzy  : accept only if filler-stripped token sets identical, number tokens identical,
         and (year gap <=1 when both known for non-dbpedia units).
noplat : resolved in clustering (s5)."""
import os, pandas as pd, numpy as np
from feats import nums, core
W = os.path.dirname(os.path.abspath(__file__))
R = pd.read_pickle(f"{W}/state/norm.pkl"); R = R[~R.junk].copy()
R["unit"] = R.source + "|" + R.name_k.fillna("") + "|" + R.platform_k.fillna("")
agg = R.groupby("unit").agg(source=("source","first"), name=("name_c","first"),
        years=("year", lambda x: sorted(set(int(v) for v in x.dropna()))),
        critic=("critic_c", lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan))
C = pd.read_pickle(f"{W}/state/unit_candidates.pkl")
def ygap(u1, u2):
    a, b = agg.at[u1, "years"], agg.at[u2, "years"]
    if agg.at[u1, "source"] == "dbpedia" or agg.at[u2, "source"] == "dbpedia" or not a or not b: return None
    return min(abs(x - y) for x in a for y in b)
out = []
for u1, u2, how in C.itertuples(index=False):
    yg = ygap(u1, u2)
    c1, c2 = agg.at[u1, "critic"], agg.at[u2, "critic"]
    if how.startswith("exact") or how == "exact":
        if yg is not None and yg >= 5 and not np.isnan(c1) and not np.isnan(c2) and abs(c1 - c2) > 3:
            out.append((u1, u2, how, 0, 0.0, f"remake contradiction dy={yg} dcritic={abs(c1-c2)}")); continue
        out.append((u1, u2, how, 1, 1.0, "exact name+platform")); continue
    if how == "fuzzy":
        n1, n2 = agg.at[u1, "name"], agg.at[u2, "name"]
        if core(n1) == core(n2) and nums(n1) == nums(n2) and len(core(n1)) > 0 and (yg is None or yg <= 1):
            out.append((u1, u2, how, 1, 0.9, "same core tokens, filler-only difference")); continue
        out.append((u1, u2, how, 0, 0.0, "fuzzy rejected")); continue
    out.append((u1, u2, how, None, None, "deferred"))
D = pd.DataFrame(out, columns=["u1","u2","how","accept","score","reason"])
D.to_pickle(f"{W}/state/unit_decisions.pkl")
print(D.groupby(["how","accept"], dropna=False).size())
print(D[(D.how=="fuzzy")&(D.accept==1)].assign(n1=lambda d: d.u1.map(agg.name), n2=lambda d: d.u2.map(agg.name))[["n1","n2","u1"]].sample(40, random_state=0).to_string())
print(D[D.reason.str.startswith("remake")].to_string())
