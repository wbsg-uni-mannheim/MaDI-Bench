"""Label-free diagnostic panel computed from the saved submission artifacts. Appends to work/diagnostics.jsonl."""
import pandas as pd, numpy as np, json, os, re, sys, datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
REV = os.environ.get('REV', 'current')
src = {'dbpedia': ('entity_uri',), 'forbes': ('forbes_url',), 'fullcontact': ('Attribute_1',)}
ids = {s: set(pd.read_csv(f'task/input/data/{s}.csv')[c[0]]) for s, c in src.items()}
m = pd.read_csv('submission/membership.csv'); f = pd.read_csv('submission/fused.csv')
co = pd.read_csv('submission/correspondences.csv'); ca = pd.read_csv('submission/blocking/candidates.csv')
schema = json.load(open('task/input/schemamatching/target_schema.json'))
cldr = set(pd.read_csv('task/input/schemamatching/CLDR_Country_Taxonomy.csv')['Country Name'])
gics = set(pd.read_csv('task/input/schemamatching/GICS_Industry_Taxonomy.csv')['Industry Name'])
D = {'rev': REV, 'time': datetime.datetime.now().isoformat(timespec='seconds'),
     'inputs': ['submission/membership.csv', 'submission/fused.csv', 'submission/correspondences.csv', 'submission/blocking/candidates.csv']}
D['coverage'] = {s: round(len(set(m[m.source == s].record_id) & ids[s]) / len(ids[s]), 4) for s in ids}
allids = set().union(*ids.values())
D['unresolved_membership_ids'] = int((~m.record_id.isin(allids)).sum())
D['duplicate_membership_rows'] = int(m.record_id.duplicated().sum())
D['unresolved_corr_ids'] = int((~co.id1.isin(allids)).sum() + (~co.id2.isin(allids)).sum())
cand = set(zip(ca.id1, ca.id2)) | set(zip(ca.id2, ca.id1))
D['corr_in_candidates'] = round(np.mean([(a, b) in cand for a, b in zip(co.id1, co.id2)]), 4) if len(co) else None
cmap = dict(zip(m.record_id, m.cluster_id))
D['corr_same_cluster'] = round(np.mean([cmap.get(a) == cmap.get(b) for a, b in zip(co.id1, co.id2)]), 4)
sz = m.groupby('cluster_id').size(); ns = m.groupby('cluster_id').source.nunique()
D['n_clusters'] = int(len(sz)); D['size_dist'] = {int(k): int(v) for k, v in sz.value_counts().sort_index().items()}
D['sources_per_cluster'] = {int(k): int(v) for k, v in ns.value_counts().sort_index().items()}
D['singleton_share'] = round((sz == 1).mean(), 4); D['max_cluster'] = int(sz.max())
D['multi_source_share_by_source'] = {s: round(m[m.source == s].cluster_id.map(ns).gt(1).mean(), 4) for s in ids}
D['fused_ids_eq_membership'] = set(f['_id']) == set(m.cluster_id) and f['_id'].is_unique
D['n_correspondences'] = int(len(co)); D['n_candidates'] = int(len(ca))
dens = {}
for a in schema['properties']:
    v = f[a] if a in f else pd.Series(dtype=object)
    dens[a] = round(v.astype(str).str.strip().replace({'nan': '', 'None': '', '<NA>': ''}).ne('').mean(), 4)
D['density'] = dens
val = {}
val['country_in_cldr'] = round(f.country.dropna().isin(cldr).mean(), 4)
val['industry_in_gics'] = round(f.industry.dropna().isin(gics).mean(), 4)
val['founded_pattern'] = round(f.founded.dropna().astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$').mean(), 4)
yr = f.founded.dropna().astype(str).str[:4].astype(int)
val['founded_in_range'] = round(((yr >= 1700) & (yr <= 2016)).mean(), 4)
val['assets_in_range'] = round(f.assets.dropna().between(0, 4e12).mean(), 4)
val['revenue_in_range'] = round(f.revenue.dropna().between(0, 1e12).mean(), 4)
val['name_len_ok'] = round(f.name.astype(str).str.len().between(1, 200).mean(), 4)
def kp_ok(v):
    try: l = json.loads(v); return isinstance(l, list) and 1 <= len(l) <= 20
    except Exception: return False
val['keypeople_json_list'] = round(f.keypeople.dropna().map(kp_ok).mean(), 4)
D['validity'] = val
# provenance trace: every fused value traced to a member record of the same cluster
pv = pd.read_csv('work/state/fusion_provenance.csv')
D['provenance_member_ok'] = round(np.mean([cmap.get(r) == c for r, c in zip(pv.record_id, pv.cluster_id)]), 4)
D['attr_disagreement_rate'] = {a: round((g.n_distinct_values > 1).mean(), 4) for a, g in pv.groupby('attribute')}
open('work/diagnostics.jsonl', 'a').write(json.dumps(D) + '\n')
print(json.dumps(D, indent=1))
