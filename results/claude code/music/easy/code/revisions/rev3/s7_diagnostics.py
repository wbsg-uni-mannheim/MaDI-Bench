import pandas as pd, numpy as np, json, re, datetime, sys
S='submission/'
rev=sys.argv[1] if len(sys.argv)>1 else 'current'
src={s:pd.read_csv(f'task/input/data/{s}.csv',dtype=str) for s in ['discogs','lastfm','musicbrainz']}
allids={i:s for s,d in src.items() for i in d.id}
mem=pd.read_csv(S+'membership.csv',dtype=str); cor=pd.read_csv(S+'correspondences.csv',dtype={'id1':str,'id2':str})
fus=pd.read_csv(S+'fused.csv',dtype=str,keep_default_na=False,na_values=['']); cand=pd.read_csv(S+'blocking/candidates.csv',dtype=str)
D={'revision':rev,'ts':datetime.datetime.now().isoformat(timespec='seconds')}
D['unresolved_member_ids']=int((~mem.record_id.isin(allids)).sum())
D['source_mismatch']=int((mem.source!=mem.record_id.map(allids)).sum())
D['coverage']={s:float(d.id.isin(mem.record_id).mean()) for s,d in src.items()}
D['dup_membership']=int(mem.record_id.duplicated().sum())
D['cluster_vs_fused']=[int((~mem.cluster_id.isin(fus._id)).sum()),int((~fus._id.isin(mem.cluster_id)).sum())]
cs=set(map(tuple,np.sort(cand[['id1','id2']].values,axis=1)))
cp=np.sort(cor[['id1','id2']].values,axis=1)
D['corr_in_candidates']=float(np.mean([tuple(x) in cs for x in cp]))
cmap=dict(zip(mem.record_id,mem.cluster_id))
D['corr_same_cluster']=float(np.mean([cmap[x]==cmap[y] for x,y in cp]))
D['corr_same_source']=int(sum(allids[x]==allids[y] for x,y in cp))
sz=mem.groupby('cluster_id').size(); D['cluster_sizes']=sz.value_counts().sort_index().to_dict(); D['singleton_share']=float((sz==1).mean())
D['max_same_source_in_cluster']=int(mem.groupby(['cluster_id','source']).size().max())
D['n_candidates']=len(cand); D['n_corr']=len(cor); D['n_fused']=len(fus)
D['density']={c:float(fus[c].notna().mean()) for c in fus.columns}
D['valid_date']=float(fus['release-date'].dropna().str.fullmatch(r'\d{4}-\d{2}-\d{2}').mean())
dd=pd.to_datetime(fus['release-date'],errors='coerce',format='%Y-%m-%d'); D['unparseable_date']=int((dd.isna()&fus['release-date'].notna()).sum())
D['duration_int']=float(fus.duration.dropna().str.fullmatch(r'\d+').mean())
D['tracks_json']=float(fus.tracks.dropna().map(lambda s: isinstance(json.loads(s),list)).mean())
# trace: fused values must come from own members (normalized) — sampled check over all multi-member clusters
nm=pd.read_pickle('work/state/normalized.pkl').set_index('id')
sys.path.insert(0,'work'); from common import key
bad=0; multi=sz[sz>1].index; fi=fus.set_index('_id')
for cid in multi:
    ids=mem.record_id[mem.cluster_id==cid]; r=fi.loc[cid]
    if pd.notna(r['name']) and key(r['name']) not in {key(x) for x in nm.loc[ids,'name']}: bad+=1
D['fused_name_not_from_members']=bad
print(json.dumps(D,indent=1))
open('work/diagnostics.jsonl','a').write(json.dumps(D)+'\n')
