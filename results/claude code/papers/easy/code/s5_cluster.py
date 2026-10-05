"""Stage 5: constrained greedy clustering. Accepted edges sorted by score (tie-break ids) are merged
only if the merged cluster keeps <=1 record per source (each source lists a publication once;
same-title records within a source are distinct versions). Unmatched records stay singletons."""
import pandas as pd, numpy as np, os, itertools, json
W=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(W); ST=f'{W}/state'
SRCS=['crossref','dblp','open_alex']
C=pd.read_pickle(f'{ST}/scored.pkl')
E=C[C.acc].sort_values(['score','id1','id2'],ascending=[False,True,True])
ids=pd.concat([pd.read_pickle(f'{ST}/norm_{s}.pkl')[['id','source']] for s in SRCS])
src=dict(zip(ids.id,ids.source))
parent={i:i for i in ids.id}; members={i:{src[i]:i} for i in ids.id}
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
rej=[]; kept=[]; rej_bridge=[]; MINT=0.8; MINS=0.7
pair_S={(a,b):v for a,b,v in zip(C.id1,C.id2,C.score)}; pair_S.update({(b,a):v for (a,b),v in list(pair_S.items())})
# implied pairs need title agreement themselves (T>=MINT) unless a record's title has <=2 tokens (truncated/acronym-only)
pair_T={(a,b):v for a,b,v in zip(C.id1,C.id2,C['T'])}; pair_T.update({(b,a):v for (a,b),v in list(pair_T.items())})
N=pd.concat([pd.read_pickle(f'{ST}/norm_{s}.pkl')[['id','tkey']] for s in SRCS])
short=dict(zip(N.id,N.tkey.str.split().str.len().fillna(0)<=2))
for x,y,s in zip(E.id1,E.id2,E.score):
    rx,ry=find(x),find(y)
    if rx==ry: kept.append((x,y,s)); continue
    if set(members[rx])&set(members[ry]): rej.append((x,y,s)); continue
    # no weak transitive bridge
    if any(((pair_T.get((p,q),-9)<MINT and not (short[p] or short[q])) or pair_S.get((p,q),9)<MINS) for p in members[rx].values() for q in members[ry].values() if (p,q)!=(x,y) and (q,p)!=(x,y)):
        rej_bridge.append((x,y,s)); continue
    if len(members[rx])<len(members[ry]): rx,ry=ry,rx
    parent[ry]=rx; members[rx].update(members.pop(ry)); kept.append((x,y,s))
print('accepted edges',len(E),'used',len(kept),'rejected by source constraint',len(rej),'rejected weak bridge',len(rej_bridge))
pd.DataFrame(rej_bridge,columns=['id1','id2','score']).to_csv(f'{ST}/rejected_bridges.csv',index=False)
pd.DataFrame(rej,columns=['id1','id2','score']).to_csv(f'{ST}/rejected_edges.csv',index=False)
# cluster ids: deterministic, based on smallest member id
roots=sorted(members, key=lambda r: min(members[r].values()))
rows=[]; cid={}
for k,r in enumerate(roots):
    c=f'e{k:06d}'
    for s,i in members[r].items(): rows.append((i,s,c))
M=pd.DataFrame(rows,columns=['record_id','source','cluster_id']).sort_values(['cluster_id','source'])
M.to_csv(f'{R}/submission/membership.csv',index=False)
# correspondences: every cross-source pair inside a cluster; score = direct candidate score if exists
sc={(a,b):s for a,b,s in zip(C.id1,C.id2,C.score)}; sc.update({(b,a):s for (a,b),s in list(sc.items())})
cor=[]
for c,g in M.groupby('cluster_id'):
    for a,b in itertools.combinations(sorted(g.record_id),2):
        cor.append((a,b,round(float(min(1.0,max(0.0,sc.get((a,b),np.nan)))) if (a,b) in sc else np.nan,4)))
K=pd.DataFrame(cor,columns=['id1','id2','score'])
print('correspondences',len(K),'implied-not-in-candidates',K.score.isna().sum())
K['score']=K.score.fillna(0.5)
K.to_csv(f'{R}/submission/correspondences.csv',index=False)
sz=M.groupby('cluster_id').size(); print('cluster size dist',sz.value_counts().sort_index().to_dict())
print('source combos',M.groupby('cluster_id').source.apply(lambda s:'+'.join(sorted(s))).value_counts().to_dict())
