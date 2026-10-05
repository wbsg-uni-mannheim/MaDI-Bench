"""Stage 5: clustering.
1. Automatic graph: accepted pairs from s4 -> connected components (fallback for unreviewed records).
2. Direct review decisions (work/manual/decisions.txt, produced by inspecting every spec block listed in
   work/manual/review_snapshot_v1.txt): each group line is a union of records asserted to be the same entity.
   Every record in a block marked '### <block>' is 'reviewed': its automatic edges are dropped and it is
   assigned only through the explicit group lines (unlisted reviewed records stay singletons)."""
import pandas as pd, networkx as nx, json
n = pd.read_csv('work/state/s2_features.csv', dtype=str, keep_default_na=False)
p = pd.read_csv('work/state/s4_scored_pairs.csv', dtype=str, keep_default_na=False)
full = {x.replace('products_',''): x for x in n.id}
snap = open('work/manual/review_snapshot_v1.txt').read().split('\n')
blocks = {}; cur = None
for l in snap:
    if l.startswith('### '): cur = l[4:].strip(); blocks[cur] = []
    elif l.strip(): blocks[cur].append(l.split()[1])
reviewed = set(); groups = []
for l in open('work/manual/decisions.txt').read().split('\n'):
    if l.startswith('### '):
        reviewed.update(full[x] for x in blocks[l[4:].strip()])
    elif l.strip() and not l.startswith('#'):
        groups.append([full[x] for x in l.split()])
G = nx.Graph(); G.add_nodes_from(n.id)
auto_kept = 0
for r in p[p.match=='1'].itertuples():
    if r.id1 in reviewed or r.id2 in reviewed: continue
    G.add_edge(r.id1, r.id2, kind='auto'); auto_kept += 1
for g in groups:
    for a, b in zip(g, g[1:]): G.add_edge(a, b, kind='manual')
src = dict(zip(n.id, n.source))
rows = []
for c in nx.connected_components(G):
    cid = 'c_' + min(c).replace('products_', '')
    for x in c: rows.append(dict(record_id=x, source=src[x], cluster_id=cid))
m = pd.DataFrame(rows).sort_values(['cluster_id', 'record_id'])
m.to_csv('work/state/s5_membership.csv', index=False)
sz = m.cluster_id.value_counts()
stats = dict(records=len(m), clusters=len(sz), reviewed_records=len(reviewed), manual_group_lines=len(groups),
             auto_edges_kept=auto_kept, size_dist={int(k): int(v) for k, v in sz.value_counts().sort_index().items()})
print(json.dumps(stats))
json.dump(stats, open('work/state/s5_stats.json', 'w'))
