"""Stage 7: export full record-level blocking candidates (+transitive closure pairs) and label-free diagnostics."""
import pandas as pd, json, datetime, sys
sys.path.insert(0,'work')
N = pd.read_pickle('work/state/clustered.pkl')
C = pd.read_pickle('work/state/candidates_units.pkl')
recs = N.groupby('unit').record_id.apply(list).to_dict()
pairs = set()
for a, b in zip(C.u1, C.u2):
    for x in recs[a]:
        for y in recs[b]: pairs.add((x, y) if x < y else (y, x))
Cr = pd.read_csv('submission/correspondences.csv')
corr = set((a,b) if a < b else (b,a) for a,b in zip(Cr.id1, Cr.id2))
trans = corr - pairs
pairs |= trans   # candidates added by transitive closure over matched units
B = pd.DataFrame(sorted(pairs), columns=['id1','id2'])
B.to_csv('submission/blocking/candidates.csv', index=False)
# ---- diagnostics (re-read saved files)
M = pd.read_csv('submission/membership.csv'); F = pd.read_csv('submission/fused.csv'); Cr = pd.read_csv('submission/correspondences.csv')
src = {s: pd.read_csv(f'task/input/data/{s}.csv', dtype=str).iloc[:,0] for s in ['dbpedia','metacritic','sales']}
allids = set().union(*[set(v) for v in src.values()])
d = {}
d['records_total'] = len(allids); d['membership_rows'] = len(M); d['membership_unique'] = M.record_id.nunique()
d['unknown_ids_membership'] = int((~M.record_id.isin(allids)).sum())
d['missing_ids_membership'] = len(allids - set(M.record_id))
d['clusters'] = int(M.cluster_id.nunique()); d['fused_rows'] = len(F)
d['cluster_ids_not_in_fused'] = len(set(M.cluster_id) - set(F._id)); d['fused_not_in_membership'] = len(set(F._id) - set(M.cluster_id))
cs = M.groupby('cluster_id').source.agg(lambda s: '+'.join(sorted(set(s))))
d['source_combo'] = cs.value_counts().to_dict()
d['singleton_share'] = round(float((M.groupby('cluster_id').size()==1).mean()),4)
d['cluster_size_max'] = int(M.groupby('cluster_id').size().max())
per = M.groupby(['cluster_id','source']).size().unstack(fill_value=0)
d['clusters_with_>1_metacritic'] = int((per.get('metacritic',0)>1).sum()); d['clusters_with_>1_sales'] = int((per.get('sales',0)>1).sum())
d['candidates'] = len(B); d['candidates_transitive_added'] = len(trans)
full = 46580*20494 + 46580*7877 + 20494*7877
d['reduction_ratio'] = round(1 - len(B)/full, 6)
d['correspondences'] = len(Cr); d['corr_same_source'] = int((Cr.id1.str.split('_').str[0]==Cr.id2.str.split('_').str[0]).sum())
cl = M.set_index('record_id').cluster_id
d['corr_cross_cluster'] = int((Cr.id1.map(cl) != Cr.id2.map(cl)).sum())
d['density'] = F.notna().mean().round(4).to_dict()
ok_esrb = {'E','E10+','T','M','AO','RP','RP-LM17'}
d['esrb_invalid'] = int((F.ESRB.notna() & ~F.ESRB.isin(ok_esrb)).sum())
d['year_pattern_invalid'] = int((F.releaseYear.notna() & ~F.releaseYear.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$')).sum())
d['critic_out_of_range'] = int(((F.criticScore<0)|(F.criticScore>100)).sum()); d['user_out_of_range'] = int(((F.userScore<0)|(F.userScore>10)).sum())
gl = F.genres.dropna().map(json.loads).map(len); d['genres_len_max'] = int(gl.max()); d['genres_over10'] = int((gl>10).sum())
P = pd.read_csv('work/state/fusion_provenance.csv')
m = P[P.sources.str.contains(r'\+')]
d['multi_src_year_conflict_rate'] = round(float((m.year_distinct>1).mean()),4); d['multi_src_dev_conflict_rate'] = round(float((m.dev_distinct>1).mean()),4)
d['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds'); d['revision'] = open('work/REVISION').read().strip()
d['inputs'] = ['submission/membership.csv','submission/fused.csv','submission/correspondences.csv','submission/blocking/candidates.csv']
open('work/diagnostics.jsonl','a').write(json.dumps(d)+'\n')
print(json.dumps(d, indent=1))
