"""Stage 5: constrained greedy agglomeration over accepted match edges.
Edges are processed strongest first (model-number equality, then shared code, then text; ties by cosine, then ids).
Two clusters are merged only if (a) no member pair has a hard conflict and (b) the merge does not put two records of the
same source together unless those two records are themselves linked by an accepted edge."""
import os, sys, json, collections
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairfeat import pair_features, code_doc_freq, set_word_df
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
d = pd.read_pickle('work/state/norm.pkl')
c = pd.read_pickle('work/state/scored.pkl')
set_word_df(d.title_n.tolist(), d.brand_key.tolist())
R = d.to_dict('records'); code_df = code_doc_freq(d)
m = c[c.match].copy()
m['prio'] = np.where(m.mn_eq, 2, np.where(m.n_shared > 0, 1, 0))
m = m.sort_values(['prio', 'cos', 'id1', 'id2'], ascending=[False, False, True, True])
edge = set(zip(m.i1, m.i2)) | set(zip(m.i2, m.i1))
cache = {}
def conflict(a, b):
    k = (min(a, b), max(a, b))
    if k not in cache: cache[k] = pair_features(R[k[0]], R[k[1]], code_df)['hard_conf']
    return cache[k]
cl = {i: i for i in range(len(d))}; members = {i: [i] for i in range(len(d))}
rejected = collections.Counter(); accepted = []
for r in m.itertuples():
    ca, cb = cl[r.i1], cl[r.i2]
    if ca == cb: accepted.append((r.i1, r.i2)); continue
    A, B = members[ca], members[cb]
    bad = None
    for x in A:
        for y in B:
            if R[x]['source'] == R[y]['source'] and (x, y) not in edge: bad = 'same_source'; break
            if conflict(x, y): bad = 'conflict'; break
        if bad: break
    if bad: rejected[bad] += 1; continue
    accepted.append((r.i1, r.i2))
    keep, drop = (ca, cb) if len(A) >= len(B) else (cb, ca)
    for x in members[drop]: cl[x] = keep
    members[keep] += members.pop(drop)
# cluster ids: deterministic, based on the smallest record id in the cluster
ids = d.id.values
cid = {}
for k, mem in members.items():
    cid[k] = 'E_' + min(ids[x] for x in mem)
d_cl = pd.DataFrame({'record_id': ids, 'source': d.source.values, 'cluster_id': [cid[cl[i]] for i in range(len(d))]})
d_cl.to_csv('work/state/membership.csv', index=False)
acc = pd.DataFrame(accepted, columns=['i1', 'i2']).merge(m[['i1', 'i2', 'id1', 'id2', 'score', 'rule']], on=['i1', 'i2'])
acc.to_pickle('work/state/accepted_edges.pkl')
# correspondences = all cross-source pairs within final clusters (transitively implied), scored by direct edge if present
sc = {(a, b): s for a, b, s in zip(m.id1, m.id2, m.score)}
rows = []
for k, mem in members.items():
    mem = sorted(ids[x] for x in mem)
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            a, b = mem[i], mem[j]
            if a.rsplit('_', 1)[0] == b.rsplit('_', 1)[0]: continue
            rows.append((a, b, sc.get((a, b), sc.get((b, a), 0.5))))
pd.DataFrame(rows, columns=['id1', 'id2', 'score']).to_csv('work/state/correspondences.csv', index=False)
sizes = pd.Series([len(v) for v in members.values()])
nsrc = pd.Series([len({R[x]['source'] for x in v}) for v in members.values()])
stats = {'clusters': len(members), 'size_dist': sizes.value_counts().sort_index().to_dict(),
         'sources_per_cluster': nsrc.value_counts().sort_index().to_dict(), 'singleton_share': float((sizes == 1).mean()),
         'rejected_merges': dict(rejected), 'accepted_edges': len(accepted), 'correspondences': len(rows),
         'clusters_with_same_source_dupes': int((sizes != nsrc).sum())}
print(json.dumps(stats)); json.dump(stats, open('work/state/cluster_stats.json', 'w'))
