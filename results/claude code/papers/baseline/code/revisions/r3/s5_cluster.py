"""Stage 5: constrained clustering. Greedy union of accepted edges by descending score;
a merge is refused if the merged cluster would hold two records of one source or contain a
pair that the matcher hard-rejected (volume/page/number/type contradiction)."""
import os, json, numpy as np, pandas as pd
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
N = pd.read_pickle('work/state/s2_normalized.pkl')
C = pd.read_pickle('work/state/s4_decisions.pkl')
src = N.source.to_numpy(); ids = N.id.to_numpy()
rej = set(zip(C.i[C.hard_reject], C.j[C.hard_reject]))
parent = list(range(len(N))); members = {i: [i] for i in range(len(N))}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
E = C[C.accept].sort_values(['score', 'i', 'j'], ascending=[False, True, True])
kept, refused = [], Counter()
for r in E.itertuples():
    a, b = find(r.i), find(r.j)
    if a == b: kept.append(r.Index); continue
    sa = {src[m] for m in members[a]}; sb = {src[m] for m in members[b]}
    if sa & sb: refused['same_source'] += 1; continue
    if any((min(x, y), max(x, y)) in rej for x in members[a] for y in members[b]):
        refused['rejected_pair'] += 1; continue
    if len(members[a]) < len(members[b]): a, b = b, a
    parent[b] = a; members[a] += members.pop(b); kept.append(r.Index)
root = np.array([find(i) for i in range(len(N))])
# cluster id = smallest source id in the cluster, deterministic
cl = pd.DataFrame({'record_id': ids, 'source': src, 'root': root})
cid = cl.groupby('root').record_id.min()
cl['cluster_id'] = 'c_' + cl.root.map(cid)
cl[['record_id', 'source', 'cluster_id']].to_csv('work/state/s5_membership.csv', index=False)
C['in_cluster'] = False; C.loc[kept, 'in_cluster'] = True
C['same_cluster'] = root[C.i] == root[C.j]
C.to_pickle('work/state/s5_edges.pkl')
sz = cl.groupby('cluster_id').size()
sig = cl.groupby('cluster_id').source.apply(lambda s: '+'.join(sorted(s))).value_counts()
diag = dict(stage='clustering', n_records=len(cl), n_clusters=int(len(sz)), size_dist=sz.value_counts().sort_index().to_dict(),
            source_signature=sig.to_dict(), refused=dict(refused), accepted_edges=int(len(E)),
            edges_within_clusters=int(C[C.accept].same_cluster.sum()),
            singleton_share=round(float((sz == 1).mean()), 4))
print(json.dumps(diag, indent=1)); json.dump(diag, open('work/state/s5_diag.json', 'w'))
