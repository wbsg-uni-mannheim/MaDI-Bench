"""Label-free diagnostic panel over the SAVED submission artifacts. Appends to work/diagnostics.jsonl."""
import pandas as pd, numpy as np, json, os, time, hashlib, re
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W); SUB = f'{ROOT}/submission'
rev = open(f'{W}/REVISION').read().strip() if os.path.exists(f'{W}/REVISION') else 'dev'
src = {f'products_{i}': pd.read_csv(f'{ROOT}/task/input/data/products_{i}.csv', dtype=str, keep_default_na=False) for i in range(1, 5)}
allids = {i: s for s, d in src.items() for i in d.id}
mem = pd.read_csv(f'{SUB}/membership.csv', dtype=str)
fused = pd.read_csv(f'{SUB}/fused.csv', dtype=str, keep_default_na=False)
corr = pd.read_csv(f'{SUB}/correspondences.csv', dtype={'id1': str, 'id2': str})
cand = pd.read_csv(f'{SUB}/blocking/candidates.csv', dtype=str)
schema = json.load(open(f'{ROOT}/task/input/schemamatching/target_schema.json'))['properties']
d = {'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), 'revision': rev,
     'inputs': ['submission/membership.csv', 'submission/fused.csv', 'submission/correspondences.csv', 'submission/blocking/candidates.csv']}
d['source_records'] = len(allids)
d['membership_rows'] = len(mem); d['membership_unique_ids'] = mem.record_id.nunique()
d['unresolved_membership_ids'] = int((~mem.record_id.isin(allids)).sum())
d['source_label_mismatch'] = int(sum(allids.get(r) != s for r, s in zip(mem.record_id, mem.source)))
d['uncovered_source_records'] = len(set(allids) - set(mem.record_id))
d['fused_rows'] = len(fused); d['fused_ids_unique'] = bool(fused._id.is_unique)
d['membership_clusters_missing_in_fused'] = len(set(mem.cluster_id) - set(fused._id))
d['fused_rows_without_members'] = len(set(fused._id) - set(mem.cluster_id))
sz = mem.groupby('cluster_id').size()
d['cluster_size_dist'] = {int(k): int(v) for k, v in sz.value_counts().sort_index().items()}
d['singleton_share'] = round(float((sz == 1).mean()), 4)
nsrc = mem.groupby('cluster_id').source.nunique()
d['sources_per_cluster_dist'] = {int(k): int(v) for k, v in nsrc.value_counts().sort_index().items()}
dup = mem.groupby(['cluster_id', 'source']).size()
d['clusters_with_same_source_duplicates'] = int((dup > 1).groupby(level=0).any().sum())
d['records_in_same_source_dup_clusters'] = int(dup[dup > 1].sum())
cm = dict(zip(mem.record_id, mem.cluster_id))
d['correspondences'] = len(corr)
d['corr_unresolved_ids'] = int((~corr.id1.isin(allids)).sum() + (~corr.id2.isin(allids)).sum())
d['corr_cross_cluster'] = int(sum(cm[a] != cm[b] for a, b in zip(corr.id1, corr.id2)))
d['corr_same_source'] = int(sum(allids[a] == allids[b] for a, b in zip(corr.id1, corr.id2)))
ck = {tuple(sorted(x)) for x in zip(cand.id1, cand.id2)}
d['corr_in_candidates'] = round(float(np.mean([tuple(sorted(x)) in ck for x in zip(corr.id1, corr.id2)])), 4)
d['candidates'] = len(cand)
N = len(allids); d['reduction_ratio'] = round(1 - len(cand) / (N * (N - 1) / 2), 5)
# schema / type validity + density
dens, invalid = {}, {}
for p, spec in schema.items():
    v = fused[p].astype(str).str.strip()
    nn = v[v != '']
    dens[p] = round(len(nn) / len(fused), 4)
    bad = 0
    if spec.get('type') == 'number':
        num = pd.to_numeric(nn, errors='coerce')
        bad = int(num.isna().sum())
        if 'minimum' in spec: bad += int((num < spec['minimum']).sum())
        if 'maximum' in spec: bad += int((num > spec['maximum']).sum())
    if 'enum' in spec: bad += int((~nn.isin(spec['enum'])).sum())
    if 'pattern' in spec: bad += int((~nn.str.match(spec['pattern'])).sum())
    if 'maxLength' in spec: bad += int((nn.str.len() > spec['maxLength']).sum())
    if bad: invalid[p] = bad
d['density'] = dens; d['schema_violations'] = invalid
inscope = fused.product_type.isin(['GPU', 'SSD', 'HDD', 'USB_STICK'])
d['fused_inscope_rows'] = int(inscope.sum()); d['fused_out_of_scope_rows'] = int((~inscope).sum())
d['sha_fused'] = hashlib.sha256(open(f'{SUB}/fused.csv', 'rb').read()).hexdigest()[:16]
d['sha_membership'] = hashlib.sha256(open(f'{SUB}/membership.csv', 'rb').read()).hexdigest()[:16]
with open(f'{W}/diagnostics.jsonl', 'a') as f: f.write(json.dumps(d) + '\n')
print(json.dumps({k: v for k, v in d.items() if k not in ('density',)}, indent=None))
print('density', d['density'])
