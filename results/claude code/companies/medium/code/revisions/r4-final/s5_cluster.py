"""Stage 5: refinement/clustering. Greedy constrained union-find over accepted edges in descending score order:
a cluster may contain at most one dbpedia and at most one forbes record (each source lists an entity once;
dbpedia URIs / forbes profiles are unique entities). fullcontact may contribute several duplicate rows.
Edges that would violate the constraint are rejected and logged."""
import pandas as pd, json
U=pd.read_pickle('work/state/s2_normalized.pkl')
M=pd.read_pickle('work/state/s4_scored.pkl')
A=M[M.accept].copy()
# within dbpedia / within forbes edges are never used (unique-per-source assumption)
A=A[~((A.s1==A.s2)&(A.s1!='fullcontact'))]
# within-fullcontact duplicates: only exact-name (N1) edges
A=A[~((A.s1==A.s2)&(A.cls!='N1'))]
A=A.sort_values(['score','id1','id2'],ascending=[False,True,True])
src=dict(zip(U.record_id,U.source))
parent={r:r for r in U.record_id}; comp={r:{'dbpedia':0,'forbes':0,'fullcontact':0} for r in U.record_id}
for r in U.record_id: comp[r][src[r]]=1
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
used=[];rejected=[]
for r in A.itertuples():
    x,y=find(r.id1),find(r.id2)
    if x==y: used.append(r.Index); continue
    cx,cy=comp[x],comp[y]
    if cx['dbpedia']+cy['dbpedia']>1 or cx['forbes']+cy['forbes']>1:
        rejected.append(dict(id1=r.id1,id2=r.id2,n1=r.n1,n2=r.n2,cls=r.cls,score=r.score,reason='one-per-source constraint')); continue
    parent[y]=x; comp[x]={k:cx[k]+cy[k] for k in cx}; used.append(r.Index)
U['root']=U.record_id.map(find)
# deterministic cluster id: first record id (sorted by source priority then id) in cluster
pri={'forbes':0,'dbpedia':1,'fullcontact':2}
U['_p']=U.source.map(pri)
first=U.sort_values(['_p','record_id']).groupby('root').record_id.first()
U['cluster_id']=U.root.map(first)
U[['record_id','source','cluster_id']].to_pickle('work/state/s5_membership.pkl')
pd.DataFrame(rejected).to_csv('work/state/s5_rejected_edges.csv',index=False)
# correspondences: all cross-source record pairs inside each final cluster (score = best direct edge score if any else implied)
E=M[M.accept].set_index(['id1','id2']).score.to_dict()
rows=[]
for cid,g in U.groupby('cluster_id'):
    ids=g.record_id.tolist()
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            a,b=ids[i],ids[j]
            if src[a]==src[b]: continue
            sc=E.get((a,b),E.get((b,a)))
            rows.append((a,b,round(min(1.0,max(0.5,(sc if sc is not None else 8)/12)),3)))
Cr=pd.DataFrame(rows,columns=['id1','id2','score'])
Cr.to_pickle('work/state/s5_correspondences.pkl')
sz=U.groupby('cluster_id').size()
print(dict(clusters=len(sz),singletons=int((sz==1).sum()),max_size=int(sz.max()),size_dist=sz.value_counts().sort_index().to_dict(),
  rejected_edges=len(rejected),correspondences=len(Cr)))
