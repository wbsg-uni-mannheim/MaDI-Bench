"""Stage 5: constrained greedy clustering: accepted edges processed by descending score; a merge is
rejected if the resulting cluster would hold two units from the same source (each source has one
unit per entity: metacritic/sales rows, dbpedia title+platform groups)."""
import pandas as pd, sys, json
P = pd.read_pickle('work/state/pair_scores.pkl')
U = pd.read_pickle('work/state/units.pkl').set_index('unit')
N = pd.read_pickle('work/state/normalized_units.pkl')
A = P[P.accept].sort_values(['score','u1','u2'], ascending=[False,True,True])
parent = {u:u for u in U.index}; members = {u:{U.source[u]:u} for u in U.index}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
kept, rejected = [], []
for r in A.itertuples():
    ra, rb = find(r.u1), find(r.u2)
    if ra == rb: kept.append(r.Index); continue
    if set(members[ra]) & set(members[rb]):
        rejected.append(dict(u1=r.u1, u2=r.u2, score=r.score, rule=r.rule, reason='source_conflict')); continue
    parent[rb] = ra; members[ra].update(members.pop(rb)); kept.append(r.Index)
pd.DataFrame(rejected).to_csv('work/state/rejected_edges.csv', index=False)
root = {u: find(u) for u in U.index}
N['croot'] = N.unit.map(root)
# cluster id: representative record id (metacritic > sales > dbpedia, then smallest numeric id)
order = {'metacritic':0,'sales':1,'dbpedia':2}
N['_o'] = N.source.map(order); N['_n'] = N.record_id.str.extract(r'_(\d+)$')[0].astype(int)
rep = N.sort_values(['_o','_n']).groupby('croot').record_id.first()
N['cluster_id'] = 'e_' + N.croot.map(rep)
N.drop(columns=['_o','_n']).to_pickle('work/state/clustered.pkl')
N[['record_id','source','cluster_id']].to_csv('submission/membership.csv', index=False)
# correspondences: all cross-source record pairs inside each cluster whose units are linked by a kept edge path
# (cluster is built only from accepted edges, so every cross-source pair within a cluster is asserted)
K = P.loc[kept]
unit_score = {}
for r in K.itertuples(): unit_score[(r.u1,r.u2)] = unit_score[(r.u2,r.u1)] = r.score
recs = N.groupby('unit').record_id.apply(list)
out = []
for cid, g in N.groupby('cluster_id'):
    units = g.unit.unique().tolist()
    for i in range(len(units)):
        for j in range(i+1, len(units)):
            ua, ub = units[i], units[j]
            sc = unit_score.get((ua,ub))
            if sc is None:  # linked transitively via third source
                sc = 0.5
            for x in recs[ua]:
                for y in recs[ub]: out.append((x, y, round(min(max(sc,0),1),4)))
Cr = pd.DataFrame(out, columns=['id1','id2','score'])
Cr.to_csv('submission/correspondences.csv', index=False)
print('kept edges', len(kept), 'rejected', len(rejected), 'clusters', N.cluster_id.nunique(), 'corr', len(Cr))
