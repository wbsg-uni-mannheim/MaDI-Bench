"""Stage 5: clustering. Union-find over accepted unit edges, exact edges first; a fuzzy edge is
rejected if it would put two units of the same source into one component (one unit per source).
Platform-less units attach to the unique same-name platformed cluster (if unique and source-compatible).
Outputs record-level membership + cross-source correspondences."""
import os, itertools, pandas as pd, numpy as np
W = os.path.dirname(os.path.abspath(__file__))
R = pd.read_pickle(f"{W}/state/norm.pkl")
D = pd.read_pickle(f"{W}/state/unit_decisions.pkl")
units = sorted(R.unit.unique())
par = {u: u for u in units}; srcs = {u: {u.split("|")[0]} for u in units}
def find(u):
    while par[u] != u: par[u] = par[par[u]]; u = par[u]
    return u
def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb: return True
    if srcs[ra] & srcs[rb]: return False
    if ra > rb: ra, rb = rb, ra
    par[rb] = ra; srcs[ra] |= srcs[rb]; return True
rejected = []
acc = D[D.accept == 1].sort_values(["score", "u1", "u2"], ascending=[False, True, True])
for u1, u2, how, sc in acc[["u1","u2","how","score"]].itertuples(index=False):
    if not union(u1, u2): rejected.append((u1, u2, how, "same-source conflict"))
# platform-less units
U = pd.Series(units); pl = U[U.str.endswith("|") & ~U.str.startswith(("junk|","row|"))]
name_of = lambda u: u.split("|")[1]
clusters_by_name = {}
for u in units:
    if u.startswith(("junk|","row|")) or u.endswith("|"): continue
    clusters_by_name.setdefault(name_of(u), set()).add(find(u))
noplat_edges = []
for u in pl:
    cs = clusters_by_name.get(name_of(u), set())
    s = u.split("|")[0]
    if len(cs) == 1:
        c = next(iter(cs))
        if s == "dbpedia" or s not in srcs[c]:
            # attach bypassing the per-source check for dbpedia (dbpedia rows are fragments of one article)
            rc = find(u); par[rc] = c; srcs[c] |= srcs[rc]; noplat_edges.append((u, c))
        else: rejected.append((u, c, "noplat", "same-source conflict"))
    elif len(cs) > 1: rejected.append((u, "|".join(sorted(cs))[:200], "noplat", "ambiguous platform"))
R["root"] = R.unit.map(find)
cid = R.groupby("root").rid.agg(lambda x: sorted(x, key=lambda i: (i.split('_')[0], int(i.split('_')[1])))[0])
R["cluster_id"] = R.root.map(cid)
R[["rid","source","unit","cluster_id"]].to_pickle(f"{W}/state/membership.pkl")
pd.DataFrame(rejected, columns=["u1","u2","how","reason"]).to_csv(f"{W}/state/rejected_edges.csv", index=False)
# correspondences: all cross-source record pairs within clusters
unit_score = {}
for u1, u2, sc in acc[["u1","u2","score"]].itertuples(index=False):
    unit_score[(u1, u2)] = unit_score[(u2, u1)] = sc
rows = []
for c, g in R.groupby("cluster_id"):
    if g.source.nunique() < 2: continue
    for (i1, s1, u1), (i2, s2, u2) in itertools.combinations(g[["rid","source","unit"]].itertuples(index=False, name=None), 2):
        if s1 != s2: rows.append((i1, i2, unit_score.get((u1, u2), 0.8)))
P = pd.DataFrame(rows, columns=["id1","id2","score"])
P.to_pickle(f"{W}/state/correspondences.pkl")
print("clusters", R.cluster_id.nunique(), "pairs", len(P), "rejected", len(rejected), "noplat attached", len(noplat_edges))
