import pandas as pd, numpy as np, json, sys
a=pd.read_pickle('work/state/normalized.pkl')
g=pd.read_pickle('work/state/scored.pkl')
CFG=json.load(open('work/state/match_cfg.json'))
std=(g.id_score>=CFG['id_min'])&(g.score>=CFG['accept'])
# alternative route: weaker artist agreement but same title and near-identical multi-track list
alt=(g.id_score>=CFG['alt_id_min'])&(g.ft>=CFG['alt_ft'])&(g.nt_min>=CFG['alt_ntr'])&(g.nset>=CFG['alt_nset'])&(g.score>=CFG['accept'])
# no issue-level evidence at all (tracks/duration/date/country unknown): demand stronger identity
noev=g.ft.isna()&g.dd.isna()&g.de.isna()&g.ce.isna()
std=std&(~noev|(g.id_score>=CFG['noev_id_min']))
e=g[std|alt].copy(); e['route']=np.where(std[std|alt],'std','alt')
# hard contradiction: both tracklists known and almost disjoint, plus not same duration
e=e[~((e.ft<0.3)&(e.nt_ratio.notna())&~(e.dd<=5))]
# mutual best per source pair (one record per source per entity); deterministic tie-break on ids
e=e.sort_values(['score','id1','id2'],ascending=[False,True,True])
# greedy global one-to-one per source pair: best-scoring edges first, each record used once per partner source
used=set(); keep=[]
for r in e.itertuples():
    k1=(r.i1,r.s2); k2=(r.i2,r.s1)
    if k1 in used or k2 in used: continue
    used.add(k1); used.add(k2); keep.append(r.Index)
mb=e.loc[keep]
# greedy union: highest-scoring edges first, never two records of the same source in one cluster
parent=list(range(len(a))); srcs=[{s} for s in a.source]
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
kept=[];rej=[]
for r in mb.sort_values(['score','id1','id2'],ascending=[False,True,True]).itertuples():
    x,y=find(r.i1),find(r.i2)
    if x==y: kept.append(r.Index); continue
    if srcs[x]&srcs[y]: rej.append(r.Index); continue
    parent[y]=x; srcs[x]|=srcs[y]; kept.append(r.Index)
root=[find(i) for i in range(len(a))]
a['root']=root
# cluster id = smallest-sorted member id (deterministic, source-native based)
cid=a.groupby('root').id.transform(lambda s: 'c_'+sorted(s)[0])
a['cluster_id']=cid
a[['id','source','cluster_id']].rename(columns={'id':'record_id'}).to_csv('submission/membership.csv',index=False)
a[['id','source','cluster_id']].to_pickle('work/state/membership.pkl')
# correspondences = all within-cluster cross-source pairs (consistent with clusters)
sc={(min(r.id1,r.id2),max(r.id1,r.id2)):r.score for r in mb.itertuples()}
rows=[]
for k,m in a.groupby('cluster_id'):
    ids=sorted(m.id)
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            rows.append((ids[i],ids[j],round(min(1.0,sc.get((ids[i],ids[j]),CFG['accept'])),4)))
pd.DataFrame(rows,columns=['id1','id2','score']).to_csv('submission/correspondences.csv',index=False)
mb.loc[rej].to_csv('work/state/rejected_edges.csv',index=False)
sz=a.groupby('cluster_id').size(); ns=a.groupby('cluster_id').source.nunique()
d=dict(edges_accepted_candidates=len(e),mutual_best=len(mb),rejected_by_constraint=len(rej),n_clusters=int(len(sz)),
       size_dist=sz.value_counts().sort_index().to_dict(),singleton_share=float((sz==1).mean()),
       n_corr=len(rows),source_combo=a.groupby('cluster_id').source.apply(lambda s:'+'.join(sorted(s))).value_counts().to_dict())
print(json.dumps(d,indent=1)); json.dump(d,open('work/state/cluster_diag.json','w'),indent=1)
