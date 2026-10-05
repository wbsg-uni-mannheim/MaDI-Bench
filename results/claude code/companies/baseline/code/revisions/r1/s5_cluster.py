"""Stage 5: refinement & clustering.
1) cross-source pairs with score>=THR; greedy 1:1 per source pair (highest score first, deterministic tie-break).
2) within-fullcontact exact duplicate rows (identical key, no attribute conflict) are merged (same profile returned twice).
   No within-dbpedia / within-forbes merges (each row is a distinct listed article / list entry).
3) union-find; any component with >1 dbpedia or >1 forbes record is split by dropping its weakest cross-source edges."""
import pandas as pd, numpy as np, os, sys, json, hashlib
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
M = pd.read_csv('work/state/scored_pairs.csv')
df = pd.read_pickle('work/state/norm.pkl').set_index('id')
THR = 0.9
rejected = []
cross = M[(M.src1 != M.src2) & (M.score >= THR)].copy()
cross['sp'] = [tuple(sorted(t)) for t in zip(cross.src1, cross.src2)]
cross = cross.sort_values(['score', 'strict_f', 'id1', 'id2'], ascending=[False, False, True, True])
kept = []
for sp, g in cross.groupby('sp', sort=False):
    used = set()
    for r in g.sort_values(['score', 'strict_f', 'id1', 'id2'], ascending=[False, False, True, True]).itertuples():
        if r.id1 in used or r.id2 in used:
            rejected.append((r.id1, r.id2, r.score, 'one_to_one')); continue
        used.add(r.id1); used.add(r.id2); kept.append(r)
E = pd.DataFrame(kept)
# fullcontact internal duplicates
fc = M[(M.src1 == 'fullcontact') & (M.src2 == 'fullcontact') & (M.tier == 'exact_or_generic')]
def noconflict(a, b):
    x, y = df.loc[a], df.loc[b]
    for c in ['country', 'city', 'founded']:
        if pd.notna(x[c]) and pd.notna(y[c]) and x[c] != y[c]: return False
    return x['name'] == y['name']
dup = [(r.id1, r.id2, r.score) for r in fc.itertuples() if noconflict(r.id1, r.id2)]
# union-find
parent = {}
def find(a):
    parent.setdefault(a, a)
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
def union(a, b): parent[find(a)] = find(b)
def build(edges):
    parent.clear()
    for i in df.index: parent[i] = i
    for a, b in edges: union(a, b)
    comp = {}
    for i in df.index: comp.setdefault(find(i), []).append(i)
    return comp
edges = [(r.id1, r.id2, r.score, 'cross') for r in E.itertuples()] + [(a, b, s, 'dup') for a, b, s in dup]
while True:
    comp = build([(a, b) for a, b, _, _ in edges])
    bad = None
    for root, mem in comp.items():
        srcs = df.loc[mem, 'source']
        if (srcs == 'dbpedia').sum() > 1 or (srcs == 'forbes').sum() > 1:
            bad = set(mem); break
    if bad is None: break
    # drop weakest cross edge inside the bad component
    inside = [e for e in edges if e[0] in bad and e[3] == 'cross']
    w = min(inside, key=lambda e: (e[2], e[0], e[1]))
    edges.remove(w); rejected.append((w[0], w[1], w[2], 'component_conflict'))
comp = build([(a, b) for a, b, _, _ in edges])
# cluster ids: deterministic, from the smallest member id
mem_rows = []
for root, mem in comp.items():
    mem = sorted(mem)
    cid = 'c_' + hashlib.md5('|'.join(mem).encode()).hexdigest()[:12]
    for m in mem: mem_rows.append((m, df.loc[m, 'source'], cid))
memb = pd.DataFrame(mem_rows, columns=['record_id', 'source', 'cluster_id'])
memb.to_csv('work/state/membership.csv', index=False)
# correspondences: all cross-source pairs inside final clusters (consistent with membership)
cmap = dict(zip(memb.record_id, memb.cluster_id))
score = {}
for a, b, s, _ in edges: score[(a, b)] = score[(b, a)] = s
corr = []
for cid, g in memb.groupby('cluster_id'):
    ids = g.record_id.tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if df.loc[a, 'source'] != df.loc[b, 'source']:
                corr.append((a, b, round(min(1.0, score.get((a, b), 0.9)), 3)))
corr = pd.DataFrame(corr, columns=['id1', 'id2', 'score'])
corr.to_csv('work/state/correspondences.csv', index=False)
pd.DataFrame(rejected, columns=['id1', 'id2', 'score', 'reason']).to_csv('work/state/rejected_edges.csv', index=False)
pd.DataFrame(edges, columns=['id1', 'id2', 'score', 'kind']).to_csv('work/state/accepted_edges.csv', index=False)
sizes = memb.groupby('cluster_id').size()
nsrc = memb.groupby('cluster_id').source.nunique()
print('clusters', len(sizes), 'size dist', sizes.value_counts().sort_index().to_dict(), 'sources/cluster', nsrc.value_counts().sort_index().to_dict())
print('correspondences', len(corr), 'rejected', pd.DataFrame(rejected, columns=['a','b','s','r']).r.value_counts().to_dict() if rejected else {})
