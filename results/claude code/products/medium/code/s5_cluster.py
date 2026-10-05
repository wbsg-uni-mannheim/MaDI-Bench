"""Stage 5: constrained greedy agglomerative clustering over accepted pairs.
Edges are processed by descending score; two clusters merge only if NO cross pair has a hard or
soft conflict (evaluated for every cross pair, candidate or not) -> no transitive bridging across
contradicting specs. Manual decisions from work/overrides.json are applied (documented in report).
Writes work/state/clusters.csv (id, source, cluster) and work/state/rejected_merges.csv."""
import pandas as pd, numpy as np, os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s4_match as M
W = M.W
n = M.n
s = pd.read_csv(f'{W}/state/scored.csv', keep_default_na=False)
OV = json.load(open(f'{W}/overrides.json')) if os.path.exists(f'{W}/overrides.json') else {'must_not': [], 'must': []}
forbid = {frozenset(p) for p in OV.get('must_not', [])}
cache = {}
def ok_pair(a, b):
    k = frozenset((a, b))
    if k in forbid: return False
    if k not in cache:
        ev = M.evidence(a, b)
        cache[k] = (ev['hard'] == '' and ev['soft'] == '')
    return cache[k]

parent = {i: i for i in n.index}; members = {i: [i] for i in n.index}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
rejected = []
def try_merge(a, b, why, force=False):
    ra, rb = find(a), find(b)
    if ra == rb: return True
    if not force:
        for x in members[ra]:
            for y in members[rb]:
                if not ok_pair(x, y):
                    rejected.append((a, b, why, x, y)); return False
    if len(members[ra]) < len(members[rb]): ra, rb = rb, ra
    parent[rb] = ra; members[ra] += members.pop(rb)
    return True

for a, b in OV.get('must', []):
    try_merge(a, b, 'override', force=True)
acc = s[s.accept == 'True'] if s.accept.dtype == object else s[s.accept]
acc = acc.sort_values(['score', 'id1', 'id2'], ascending=[False, True, True])
for r in acc.itertuples():
    try_merge(r.id1, r.id2, r.score)
roots = {}
rows = []
for i in n.index:
    rt = find(i)
    rows.append((i, n.loc[i, 'source'], rt))
cl = pd.DataFrame(rows, columns=['id', 'source', 'root'])
# deterministic cluster id: smallest member id
cid = cl.groupby('root').id.min()
cl['cluster'] = 'c_' + cl.root.map(cid)
cl[['id', 'source', 'cluster']].to_csv(f'{W}/state/clusters.csv', index=False)
pd.DataFrame(rejected, columns=['id1', 'id2', 'edge_score', 'blocking_x', 'blocking_y']).to_csv(f'{W}/state/rejected_merges.csv', index=False)
sz = cl.groupby('cluster').size()
print('clusters', len(sz), 'size dist', sz.value_counts().sort_index().to_dict(), 'rejected merges', len(rejected))
