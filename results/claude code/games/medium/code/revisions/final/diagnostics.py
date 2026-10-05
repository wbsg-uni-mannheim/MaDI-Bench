"""Label-free diagnostic panel computed from the saved submission artifacts (re-read from disk)."""
import pandas as pd, json, os, sys, datetime, collections, re
BASE = os.path.dirname(os.path.abspath(__file__)) + '/..'
S = f'{BASE}/submission'; rev = sys.argv[1] if len(sys.argv) > 1 else 'current'
src = {s: pd.read_csv(f'{BASE}/task/input/data/{s}.csv', dtype=str).id for s in ('dbpedia', 'metacritic', 'sales')}
allids = {i: s for s, ids in src.items() for i in ids}
M = pd.read_csv(f'{S}/membership.csv', dtype=str); F = pd.read_csv(f'{S}/fused.csv', dtype=str, keep_default_na=False, na_values=[''])
C = pd.read_csv(f'{S}/correspondences.csv', dtype={'id1': str, 'id2': str}); B = pd.read_csv(f'{S}/blocking/candidates.csv', dtype=str)
d = {'ts': datetime.datetime.now().isoformat(), 'revision': rev, 'inputs': ['submission/membership.csv', 'submission/fused.csv', 'submission/correspondences.csv', 'submission/blocking/candidates.csv']}
d['unresolved_membership_ids'] = int((~M.record_id.isin(allids)).sum())
d['source_label_mismatch'] = int((M.record_id.map(allids) != M.source).sum())
d['coverage_by_source'] = {s: round(M[M.source == s].record_id.nunique() / len(ids), 4) for s, ids in src.items()}
d['dup_membership'] = int(M.record_id.duplicated().sum())
d['cluster_ids_not_in_fused'] = int((~M.cluster_id.isin(F._id)).sum()); d['fused_ids_not_in_membership'] = int((~F._id.isin(M.cluster_id)).sum())
cs = M.groupby('cluster_id').agg(n=('record_id', 'size'), ns=('source', 'nunique'))
d['clusters'] = len(cs); d['singleton_share'] = round((cs.n == 1).mean(), 4); d['max_cluster'] = int(cs.n.max())
d['clusters_by_n_sources'] = cs.ns.value_counts().sort_index().to_dict()
d['clusters_with_same_source_dup'] = int((cs.n > cs.ns).sum())
sig = M.groupby('cluster_id').source.agg(lambda s: '+'.join(sorted(s))).value_counts().to_dict(); d['source_combinations'] = sig
bp = set(map(frozenset, zip(B.id1, B.id2))); cp = list(map(frozenset, zip(C.id1, C.id2)))
d['candidates'] = len(B); d['correspondences'] = len(C); d['corr_in_candidates'] = round(sum(p in bp for p in cp) / max(1, len(cp)), 4)
cl = dict(zip(M.record_id, M.cluster_id)); d['corr_same_cluster'] = round(sum(cl[a] == cl[b] for a, b in zip(C.id1, C.id2)) / max(1, len(C)), 4)
d['corr_same_source'] = int(sum(allids[a] == allids[b] for a, b in zip(C.id1, C.id2)))
tot = sum(len(v) for v in src.values()); cross = sum(len(src[a]) * len(src[b]) for a, b in [('dbpedia', 'metacritic'), ('dbpedia', 'sales'), ('metacritic', 'sales')])
d['reduction_ratio'] = round(1 - len(B) / cross, 6)
d['density'] = F.drop(columns=['_id']).notna().mean().round(4).to_dict()
esrb = set(pd.read_csv(f'{BASE}/task/input/schemamatching/ESRB_Rating_Taxonomy.csv')['Rating Code'])
v = {}
v['ESRB_in_taxonomy'] = round(F.ESRB.dropna().isin(esrb).mean(), 4)
v['date_pattern_ok'] = round(F.releaseYear.dropna().str.fullmatch(r'\d{4}-\d{2}-\d{2}').mean(), 4)
v['date_in_range'] = round(F.releaseYear.dropna().between('1960-01-01', '2024-12-31').mean(), 4)
cs_ = pd.to_numeric(F.criticScore, errors='coerce'); us = pd.to_numeric(F.userScore, errors='coerce')
v['critic_int_0_100'] = round(((cs_ % 1 == 0) & cs_.between(0, 100))[cs_.notna()].mean(), 4)
v['user_0_10'] = round(us.dropna().between(0, 10).mean(), 4)
gl = F.genres.dropna().map(json.loads); v['genres_1_10_items'] = round(gl.map(lambda l: 1 <= len(l) <= 10).mean(), 4)
plat = set(pd.read_csv(f'{BASE}/task/input/schemamatching/Gaming_Platforms_Taxonomy.csv')['Platform Name'])
v['platform_in_taxonomy_nonexhaustive'] = round(F.platform.dropna().isin(plat).mean(), 4)
for c, mx in (('name', 200), ('developer', 160), ('publisher', 160), ('series', 160), ('platform', 120)):
    v[f'{c}_len_ok'] = round(F[c].dropna().str.len().between(1, mx).mean(), 4)
d['validity'] = v
print(json.dumps(d, indent=1))
with open(f'{BASE}/work/diagnostics.jsonl', 'a') as f: f.write(json.dumps(d) + '\n')
