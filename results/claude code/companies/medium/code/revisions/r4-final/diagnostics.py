"""Label-free diagnostic panel computed from the saved submission files (not in-memory frames)."""
import pandas as pd, json, re, datetime, sys
S='submission'
src={}
for s in ['dbpedia','forbes','fullcontact']:
    d=pd.read_csv(f'task/input/data/{s}.csv',dtype=str); src.update({i:s for i in d.id})
m=pd.read_csv(f'{S}/membership.csv',dtype=str); F=pd.read_csv(f'{S}/fused.csv',dtype=str)
C=pd.read_csv(f'{S}/correspondences.csv',dtype=str); B=pd.read_csv(f'{S}/blocking/candidates.csv',dtype=str)
g=pd.read_csv('task/input/schemamatching/GICS_Industry_Taxonomy.csv',dtype=str); IND=set(g['Industry Name'])
t=pd.read_csv('task/input/schemamatching/CLDR_Country_Taxonomy.csv',dtype=str); CTRY=set(t['Country Name'])
d={}
d['source_records']=len(src); d['membership_rows']=len(m)
d['unresolved_membership_ids']=int((~m.record_id.isin(src)).sum())
d['records_missing_from_membership']=len(set(src)-set(m.record_id))
d['membership_source_mismatch']=int((m.source!=m.record_id.map(src)).sum())
d['fused_rows']=len(F); d['fused_id_unique']=bool(F._id.is_unique)
d['cluster_ids_not_in_fused']=len(set(m.cluster_id)-set(F._id)); d['fused_ids_not_in_membership']=len(set(F._id)-set(m.cluster_id))
cm=dict(zip(m.record_id,m.cluster_id))
d['correspondence_unresolved']=int((~C.id1.isin(src)|~C.id2.isin(src)).sum())
d['correspondence_same_source']=int((C.id1.map(src)==C.id2.map(src)).sum())
d['correspondence_cross_cluster']=int((C.id1.map(cm)!=C.id2.map(cm)).sum())
bk=set(map(frozenset,zip(B.id1,B.id2)))
d['correspondences_in_candidates']=round(sum(frozenset(p) in bk for p in zip(C.id1,C.id2))/max(1,len(C)),4)
d['candidates']=len(B)
sz=m.groupby('cluster_id').size(); ns=m.groupby('cluster_id').source.nunique()
d['cluster_size_dist']={int(k):int(v) for k,v in sz.value_counts().sort_index().items()}
d['sources_per_cluster']={int(k):int(v) for k,v in ns.value_counts().sort_index().items()}
d['singleton_share']=round(float((sz==1).mean()),4)
mult=m.groupby(['cluster_id','source']).size()
d['clusters_with_multiple_dbpedia_or_forbes']=int(mult[(mult>1)&(mult.index.get_level_values(1)!='fullcontact')].shape[0])
d['per_source_in_multisource_clusters']={s:int(m[(m.source==s)&m.cluster_id.map(ns).gt(1)].shape[0]) for s in ['dbpedia','forbes','fullcontact']}
dens={}
for c in F.columns:
    v=F[c].fillna('').astype(str).str.strip(); dens[c]=round(float((v!='').mean()),4)
d['density']=dens
val={}
val['country_in_taxonomy']=float(F.country.dropna().isin(CTRY).mean())
val['industry_in_taxonomy']=float(F.industry.dropna().isin(IND).mean())
fd=F.founded.dropna(); val['founded_format']=float(fd.str.fullmatch(r'\d{4}-\d{2}-\d{2}').mean()); val['founded_in_range']=float(fd.between('1700-01-01','2016-12-31').mean())
for a,lim in [('assets',4e12),('revenue',1e12)]:
    x=pd.to_numeric(F[a],errors='coerce'); val[a+'_integer_in_range']=float(((x>=0)&(x<=lim)&(x%1==0))[F[a].notna()].mean())
kp=F.keypeople.dropna(); val['keypeople_json_list']=float(kp.map(lambda s: isinstance(json.loads(s),list) and len(json.loads(s))>=1).mean())
val['name_len_ok']=float(F.name.str.len().between(1,200).mean())
d['validity']={k:round(v,4) for k,v in val.items()}
d['timestamp']=datetime.datetime.now().isoformat(timespec='seconds'); d['inputs']=S
d['revision']=sys.argv[1] if len(sys.argv)>1 else 'current'
print(json.dumps(d,indent=1))
open('work/diagnostics.jsonl','a').write(json.dumps(d)+'\n')
