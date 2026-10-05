"""Expand unit-level blocking candidates to record-level id pairs (complete set), plus all within-cluster
cross-source pairs so every correspondence is contained in the candidate file."""
import os, pandas as pd, numpy as np
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
R = pd.read_pickle(f"{W}/state/norm.pkl"); R = R[~R.junk & ~R.collection_page].copy()
ids = R.groupby("unit").rid.apply(list)
C = pd.read_pickle(f"{W}/state/unit_candidates.pkl")
rows = set()
for u1, u2 in zip(C.u1, C.u2):
    for a in ids[u1]:
        for b in ids[u2]: rows.add((a, b) if a < b else (b, a))
P = pd.read_pickle(f"{W}/state/correspondences.pkl")
extra = 0
for a, b in zip(P.id1, P.id2):
    t = (a, b) if a < b else (b, a)
    if t not in rows: rows.add(t); extra += 1
os.makedirs(f"{ROOT}/submission/blocking", exist_ok=True)
pd.DataFrame(sorted(rows), columns=["id1","id2"]).to_csv(f"{ROOT}/submission/blocking/candidates.csv", index=False)
print("candidate pairs", len(rows), "added from clustering (noplat attach)", extra)
