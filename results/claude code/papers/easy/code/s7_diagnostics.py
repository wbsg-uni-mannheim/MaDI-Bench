"""Label-free diagnostics over saved submission artifacts; appends to work/diagnostics.jsonl."""
import pandas as pd, numpy as np, os, json, time, sys, re
W=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(W); SUB=f'{R}/submission'
rev=sys.argv[1] if len(sys.argv)>1 else 'current'
src={s:set(pd.read_csv(f'{R}/task/input/data/{s}.csv',usecols=['id'],dtype=str).id) for s in ['crossref','dblp','open_alex']}
allids=set().union(*src.values())
M=pd.read_csv(f'{SUB}/membership.csv',dtype=str); K=pd.read_csv(f'{SUB}/correspondences.csv',dtype={'id1':str,'id2':str})
F=pd.read_csv(f'{SUB}/fused.csv',dtype=str); B=pd.read_csv(f'{SUB}/blocking/candidates.csv',dtype=str)
d={}
d['unresolved_membership_ids']=int((~M.record_id.isin(allids)).sum())
d['source_coverage']={s:round(len(set(M.record_id)&ids)/len(ids),4) for s,ids in src.items()}
d['dup_membership']=int(M.record_id.duplicated().sum())
d['fused_ids_eq_clusters']=set(F._id)==set(M.cluster_id) and F._id.is_unique
bs=set(zip(B.id1,B.id2))|set(zip(B.id2,B.id1))
d['candidates']=len(B); d['correspondences']=len(K)
d['corr_in_candidates']=round(np.mean([(a,b) in bs for a,b in zip(K.id1,K.id2)]),5)
cl=dict(zip(M.record_id,M.cluster_id)); d['corr_same_cluster']=bool(all(cl[a]==cl[b] for a,b in zip(K.id1,K.id2)))
sz=M.groupby('cluster_id').size(); d['cluster_size_dist']={int(k):int(v) for k,v in sz.value_counts().sort_index().items()}
d['singleton_share']=round((sz==1).mean(),4)
d['max_per_source_in_cluster']=int(M.groupby(['cluster_id','source']).size().max())
d['source_combos']=M.groupby('cluster_id').source.apply(lambda s:'+'.join(sorted(s))).value_counts().to_dict()
d['density']={c:round(F[c].fillna('').str.strip().ne('').mean(),4) for c in F.columns}
enum=json.load(open(f'{R}/task/input/schemamatching/target_schema.json'))['properties']['type']['enum']
d['type_invalid']=int((F.type.notna()&~F.type.isin(enum)).sum())
y=pd.to_numeric(F.publication_year,errors='coerce'); d['year_invalid']=int((y.notna()&~y.between(2018,2020)).sum())
# trace check: fused title/year must come from own members
N=pd.concat([pd.read_pickle(f'{W}/state/norm_{s}.pkl') for s in src]).set_index('id')
samp=F.sample(2000,random_state=0); bad=0
mem=M.groupby('cluster_id').record_id.apply(list)
for _,r in samp.iterrows():
    if isinstance(r.title,str) and r.title not in set(N.loc[mem[r._id],'title']): bad+=1
d['trace_title_not_from_members_of_2000']=bad
d.update(rev=rev,ts=time.strftime('%FT%T'),inputs='submission/*.csv')
print(json.dumps(d,indent=1))
open(f'{W}/diagnostics.jsonl','a').write(json.dumps(d)+'\n')
