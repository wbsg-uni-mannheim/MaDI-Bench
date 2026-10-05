"""Stage 5: constrained clustering.
Phase 1: accepted known-platform edges, best score first, union only if the component keeps
         <=1 metacritic record, <=1 sales record and <=1 dbpedia unit (one entity per source).
Phase 2: metacritic/sales records with unknown platform attach to one existing component when the evidence
         clearly singles it out (see UNK rules); otherwise they stay singletons.
Outputs: state/membership.csv (record_id,source,cluster_id), state/correspondences.csv, state/rejected_edges.csv
"""
import os, json, time, collections
import pandas as pd

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
UNK = dict(min_ts=0.88, strong_ev=2.0, weak_ev=1.0, margin=1.0)

ps = pd.read_csv(f"{W}/state/pair_scores.csv", keep_default_na=False)
units = pd.read_csv(f"{W}/state/dbp_units.csv", dtype=str, keep_default_na=False)
L = {s: pd.read_csv(f"{W}/state/norm_{s}.csv", dtype=str, keep_default_na=False)
     for s in ["dbpedia", "metacritic", "sales"]}

parent, comp_cnt = {}, {}


def src_of(n):
    return "dbpedia" if n.startswith("U:") else n.split("_")[0]


def add(n):
    if n not in parent:
        parent[n] = n
        comp_cnt[n] = collections.Counter({src_of(n): 1})


def find(n):
    while parent[n] != n:
        parent[n] = parent[parent[n]]
        n = parent[n]
    return n


def try_union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb:
        return True
    c = comp_cnt[ra] + comp_cnt[rb]
    if max(c.values()) > 1:
        return False
    parent[rb] = ra
    comp_cnt[ra] = c
    return True


for s in ["metacritic", "sales"]:
    for r in L[s].itertuples(index=False):
        add(r.id)
for u in set(units.unit) - {""}:
    add(u)

# phase 1
rejected = []
e1 = ps[(ps.accept == 1) & (ps.unknown_platform == 0)].sort_values(["score", "n1", "n2"], ascending=[False, True, True])
used_edges = []
for r in e1.itertuples(index=False):
    if try_union(r.n1, r.n2):
        used_edges.append((r.n1, r.n2, r.score))
    else:
        rejected.append((r.n1, r.n2, r.score, "source_constraint"))
print("phase1 edges used", len(used_edges), "rejected by constraint", len(rejected))

# phase 2: unknown-platform metacritic/sales nodes
nodes_known_pkey = {}
for s in ["metacritic", "sales"]:
    for r in L[s].itertuples(index=False):
        nodes_known_pkey[r.id] = r.pkey
e2 = ps[(ps.unknown_platform == 1) & (ps.ts >= UNK["min_ts"])]
cands = collections.defaultdict(list)
for r in e2.itertuples(index=False):
    for x, y in ((r.n1, r.n2), (r.n2, r.n1)):
        if not x.startswith("U:") and nodes_known_pkey.get(x, "x") == "":
            cands[x].append((y, r.ev, r.ts, r.score))
decisions = []
for x in sorted(cands, key=lambda k: -max(c[3] for c in cands[k])):
    rx = find(x)
    if sum(comp_cnt[rx].values()) > 1:
        continue
    by_comp = {}
    sx = src_of(x)
    for y, ev, ts, sc in cands[x]:
        ry = find(y)
        if comp_cnt[ry][sx] >= 1:
            continue
        # component-level evidence: best edge from x into that component
        if ry not in by_comp or (ev, ts) > by_comp[ry][:2]:
            by_comp[ry] = (ev, ts, y, sc)
    if not by_comp:
        continue
    ranked = sorted(by_comp.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
    best_ev = ranked[0][1][0]
    second = ranked[1][1][0] if len(ranked) > 1 else None
    ok = False
    if best_ev >= UNK["strong_ev"] and (second is None or best_ev - second >= UNK["margin"]):
        ok = True
    elif best_ev >= UNK["weak_ev"] and second is None:
        ok = True
    if ok and try_union(x, ranked[0][1][2]):
        used_edges.append((x, ranked[0][1][2], ranked[0][1][3]))
        decisions.append((x, ranked[0][1][2], best_ev, second, "attached"))
    else:
        decisions.append((x, ranked[0][1][2], best_ev, second, "left_singleton"))
dec = pd.DataFrame(decisions, columns=["node", "best_target", "best_ev", "second_ev", "decision"])
dec.to_csv(f"{W}/state/unknown_platform_decisions.csv", index=False)
print("phase2", dec.decision.value_counts().to_dict())
pd.DataFrame(rejected, columns=["n1", "n2", "score", "reason"]).to_csv(f"{W}/state/rejected_edges.csv", index=False)

# ------------------------------------------------------------------ record membership
rec_node = {}
for s in ["metacritic", "sales"]:
    for r in L[s].itertuples(index=False):
        rec_node[r.id] = r.id
for r in units.itertuples(index=False):
    if r.unit:
        rec_node[r.id] = r.unit
rows = []
for s in ["dbpedia", "metacritic", "sales"]:
    for rid in L[s].id:
        n = rec_node.get(rid)
        key = find(n) if n else "R:" + rid
        rows.append((rid, s, key))
mem = pd.DataFrame(rows, columns=["record_id", "source", "key"])
# deterministic cluster ids: ordered by the smallest member id (natural sort)
def natkey(r):
    s, n = r.rsplit("_", 1)
    return (s, int(n))
first = mem.groupby("key").record_id.agg(lambda x: min(x, key=natkey))
order = sorted(first.index, key=lambda k: natkey(first[k]))
cid = {k: f"G{i:06d}" for i, k in enumerate(order)}
mem["cluster_id"] = mem.key.map(cid)
mem[["record_id", "source", "cluster_id"]].to_csv(f"{W}/state/membership.csv", index=False)

# correspondences: all cross-source record pairs inside a cluster; score = best direct node-edge score
edge_score = {}
for a, b, sc in used_edges:
    edge_score[frozenset((find(a), a, b))] = sc
node_edge = {}
for a, b, sc in used_edges:
    node_edge[frozenset((a, b))] = sc
corr = []
for c, g in mem.groupby("cluster_id"):
    if g.source.nunique() < 2:
        continue
    recs = list(zip(g.record_id, g.source))
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            (r1, s1), (r2, s2) = recs[i], recs[j]
            if s1 == s2:
                continue
            sc = node_edge.get(frozenset((rec_node[r1], rec_node[r2])), 0.9)
            corr.append((r1, r2, round(min(1.0, sc), 4)))
cdf = pd.DataFrame(corr, columns=["id1", "id2", "score"])
cdf.to_csv(f"{W}/state/correspondences.csv", index=False)
print("clusters", mem.cluster_id.nunique(), "correspondences", len(cdf))
json.dump(UNK, open(f"{W}/state/cluster_params.json", "w"))
