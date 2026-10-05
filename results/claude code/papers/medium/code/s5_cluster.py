"""Stage 5: constrained greedy clustering. Accepted 1:1 edges are processed by descending score
(tie: ids); two clusters merge only if the union still has at most one record per source.
Edges that would violate the constraint are rejected and logged. Unmatched records stay singletons."""
import pandas as pd, networkx as nx
u = pd.read_pickle('work/state/s2_normalized.pkl')
m = pd.read_pickle('work/state/s4_matches.pkl')
src = dict(zip(u.id, u.source))
parent = {i: i for i in u.id}; members = {i: {src[i]: i} for i in u.id}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
G = nx.Graph(); G.add_nodes_from(u.id)
removed = []
edges = sorted(((s, min(a, b), max(a, b)) for a, b, s in zip(m.id1, m.id2, m.score)), key=lambda e: (-e[0], e[1], e[2]))
for s, a, b in edges:
    ra, rb = find(a), find(b)
    if ra == rb: G.add_edge(a, b, score=s); continue
    if set(members[ra]) & set(members[rb]):
        removed.append((a, b, s)); continue
    parent[rb] = ra; members[ra].update(members.pop(rb)); G.add_edge(a, b, score=s)
comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: c[0])
rows = []
for i, c in enumerate(comps):
    cid = f'e{i:06d}'
    for r in c: rows.append((r, src[r], cid))
mem = pd.DataFrame(rows, columns=['record_id', 'source', 'cluster_id'])
mem.to_pickle('work/state/s5_membership.pkl')
# correspondences: every cross-source pair inside a final cluster; score = direct candidate score if present
sc = pd.read_pickle('work/state/s4_scores.pkl').set_index(['id1', 'id2']).score.to_dict()
cor = []
for c in comps:
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            a, b = c[i], c[j]
            cor.append((a, b, sc.get((a, b), sc.get((b, a))), G.has_edge(a, b)))
cor = pd.DataFrame(cor, columns=['id1', 'id2', 'score', 'direct'])
cor.to_pickle('work/state/s5_correspondences.pkl')
pd.DataFrame(removed, columns=['id1', 'id2', 'score']).to_csv('work/state/s5_rejected_edges.csv', index=False)
sz = mem.groupby('cluster_id').size()
print('clusters', len(comps), 'size dist', sz.value_counts().sort_index().to_dict(), 'rejected edges', len(removed))
print('corr', len(cor), 'direct', int(cor.direct.sum()), 'no candidate score', int(cor.score.isna().sum()))
