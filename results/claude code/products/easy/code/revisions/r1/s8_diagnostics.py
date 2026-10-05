"""Stage 8: label-free diagnostics computed from the saved submission artifacts (re-read from disk)."""
import os, json, re, datetime
import pandas as pd, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
schema = json.load(open('task/input/schemamatching/target_schema.json'))
ATTRS = [a for a in schema['properties'] if a != 'id']
src = pd.concat([pd.read_csv(f'task/input/data/products_{i}.csv', dtype=str).assign(source=f'products_{i}') for i in range(1, 5)])
mem = pd.read_csv('submission/membership.csv', dtype=str)
cor = pd.read_csv('submission/correspondences.csv', dtype=str)
cand = pd.read_csv('submission/blocking/candidates.csv', dtype=str)
fused = pd.read_csv('submission/fused.csv', dtype=str)
diag = {'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'revision': open('work/REVISION').read().strip() if os.path.exists('work/REVISION') else None,
        'inputs': ['submission/membership.csv', 'submission/correspondences.csv', 'submission/blocking/candidates.csv', 'submission/fused.csv']}
ids = set(src.id)
diag['source_records'] = len(src)
diag['membership_rows'] = len(mem); diag['membership_unique_records'] = mem.record_id.nunique()
diag['records_missing_from_membership'] = len(ids - set(mem.record_id))
diag['unresolved_ids_membership'] = len(set(mem.record_id) - ids)
diag['unresolved_ids_corr'] = len((set(cor.id1) | set(cor.id2)) - ids)
diag['membership_source_consistent'] = bool((mem.merge(src[['id', 'source']], left_on='record_id', right_on='id').pipe(lambda x: (x.source_x == x.source_y).all())))
cs = set(map(tuple, np.sort(cand[['id1', 'id2']].values, axis=1)))
cp = set(map(tuple, np.sort(cor[['id1', 'id2']].values, axis=1)))
diag['candidates'] = len(cs); diag['correspondences'] = len(cp)
diag['correspondences_in_candidates'] = len(cp & cs)
diag['correspondences_same_source'] = int(sum(a.rsplit('_', 1)[0] == b.rsplit('_', 1)[0] for a, b in cp))
cl = dict(zip(mem.record_id, mem.cluster_id))
diag['correspondences_within_one_cluster'] = int(sum(cl[a] == cl[b] for a, b in cp))
sizes = mem.cluster_id.value_counts()
nsrc = mem.groupby('cluster_id').source.nunique()
diag['clusters'] = len(sizes); diag['size_dist'] = {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()}
diag['sources_per_cluster'] = {int(k): int(v) for k, v in nsrc.value_counts().sort_index().items()}
diag['singleton_share'] = float((sizes == 1).mean())
diag['clusters_with_same_source_duplicates'] = int((sizes.sort_index() != nsrc.sort_index()).sum())
diag['fused_rows'] = len(fused); diag['fused_ids_equal_clusters'] = set(fused._id) == set(mem.cluster_id)
diag['fused_header_ok'] = list(fused.columns) == ['_id'] + ATTRS
diag['product_type_in_enum'] = float(fused.product_type.isin(schema['properties']['product_type']['enum']).mean())
diag['density'] = fused[ATTRS].notna().mean().round(3).to_dict()
# schema checks
bad = {}
for a, p in schema['properties'].items():
    if a not in fused: continue
    v = fused[a].dropna()
    if p.get('type') == 'number':
        x = pd.to_numeric(v, errors='coerce'); n = int(x.isna().sum())
        if 'minimum' in p: n += int((x < p['minimum']).sum())
        if 'maximum' in p: n += int((x > p['maximum']).sum())
    else:
        n = 0
        if 'pattern' in p: n += int((~v.str.match(p['pattern'])).sum())
        if 'maxLength' in p: n += int((v.str.len() > p['maxLength']).sum())
        if 'enum' in p: n += int((~v.isin(p['enum'])).sum())
    if n: bad[a] = n
diag['schema_violations'] = bad
# traceability: every fused brand/model_number value is attested (case-insensitively) among its own members' raw values
s2 = src.set_index('id')
trace = {}
mcl = mem.groupby('cluster_id').record_id.apply(list)
for a in ['brand', 'model', 'model_number', 'chipset_name', 'bus_type', 'color', 'form_factor']:
    ok = tot = 0
    for r in fused[['_id', a]].dropna().itertuples():
        vals = {re.sub(r'\s', '', str(x)).lower() for x in s2.loc[mcl[r._1], a].dropna()}
        tot += 1; ok += re.sub(r'\s', '', str(r[2])).lower() in vals
    trace[a] = round(ok / tot, 4) if tot else None
diag['fused_value_attested_in_members'] = trace
# taxonomy coverage (non-exhaustive vocabularies -> coverage is descriptive only)
mem_tax = set(pd.read_csv('task/input/schemamatching/GPU_Memory_Taxonomy.csv').Variant.str.lower())
bus_tax = pd.read_csv('task/input/schemamatching/Storage_Interface_Taxonomy.csv')
bus_vocab = set(bus_tax.Variant.str.lower()) | set(bus_tax.Generation.str.lower()) | set(bus_tax['Interface Family'].str.lower())
diag['memory_type_in_taxonomy'] = float(fused.memory_type.dropna().str.lower().isin(mem_tax).mean())
diag['bus_type_in_taxonomy'] = float(fused.bus_type.dropna().str.lower().isin(bus_vocab).mean())
with open('work/diagnostics.jsonl', 'a') as f: f.write(json.dumps(diag) + '\n')
print(json.dumps({k: diag[k] for k in diag if k not in ('density',)}, indent=0)[:3000])
