"""Stage 5: connected components over 1:1 matches; split components with >1 record per source by
removing the weakest edges (deterministic). All records retained (unmatched -> singleton)."""
import pandas as pd, networkx as nx, json
u = pd.read_pickle('work/state/s2_normalized.pkl')
m = pd.read_pickle('work/state/s4_matches.pkl')
src = dict(zip(u.id, u.source))
G = nx.Graph(); G.add_nodes_from(u.id)
for a, b, s in zip(m.id1, m.id2, m.score): G.add_edge(a, b, score=s)
removed = []
def bad(comp): 
    ss = [src[n] for n in comp]; return len(ss) != len(set(ss))
changed = True
while changed:
    changed = False
    for comp in sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: c[0]):
        if len(comp) > 1 and bad(comp):
            e = min(((min(a, b), max(a, b), d) for a, b, d in G.subgraph(comp).edges(data=True)), key=lambda x: (x[2]['score'], x[0], x[1]))
            G.remove_edge(e[0], e[1]); removed.append((e[0], e[1], e[2]['score'])); changed = True
comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: c[0])
rows = []
for i, c in enumerate(comps):
    cid = f'e{i:06d}'
    for r in c: rows.append((r, src[r], cid))
mem = pd.DataFrame(rows, columns=['record_id', 'source', 'cluster_id'])
mem.to_pickle('work/state/s5_membership.pkl')
# correspondences: every cross-source pair inside a final cluster; score = direct edge score if present
sc = pd.read_pickle('work/state/s4_scores.pkl').set_index(['id1', 'id2']).score.to_dict()
cor = []
for c in comps:
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            a, b = c[i], c[j]
            s = sc.get((a, b), sc.get((b, a)))
            direct = G.has_edge(a, b)
            cor.append((a, b, s, direct))
cor = pd.DataFrame(cor, columns=['id1', 'id2', 'score', 'direct'])
cor.to_pickle('work/state/s5_correspondences.pkl')
pd.DataFrame(removed, columns=['id1', 'id2', 'score']).to_csv('work/state/s5_rejected_edges.csv', index=False)
sz = mem.groupby('cluster_id').size()
print('clusters', len(comps), 'size dist', sz.value_counts().sort_index().to_dict(), 'removed edges', len(removed))
print('corr', len(cor), 'direct', cor.direct.sum(), 'no candidate score', cor.score.isna().sum())
