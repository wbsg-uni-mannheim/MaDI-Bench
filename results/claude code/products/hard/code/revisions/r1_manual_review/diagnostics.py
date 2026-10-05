"""Label-free diagnostic panel computed from saved submission artifacts. Appends to work/diagnostics.jsonl."""
import pandas as pd, json, re, sys, datetime, itertools
rev = sys.argv[1] if len(sys.argv) > 1 else 'unnamed'
srcs = {f'products_{i}': pd.read_csv(f'task/input/data/products_{i}.csv', dtype=str) for i in range(1, 5)}
ids = {s: set(d.id) for s, d in srcs.items()}
allids = set().union(*ids.values())
m = pd.read_csv('submission/membership.csv', dtype=str)
f = pd.read_csv('submission/fused.csv', dtype=str, keep_default_na=False)
c = pd.read_csv('submission/correspondences.csv', dtype=str)
b = pd.read_csv('submission/blocking/candidates.csv', dtype=str)
sch = json.load(open('task/input/schemamatching/target_schema.json'))['properties']
D = {}
D['membership_rows'] = len(m); D['unresolved_ids'] = int((~m.record_id.isin(allids)).sum())
D['source_coverage'] = {s: round(len(set(m.record_id) & v) / len(v), 4) for s, v in ids.items()}
D['source_label_mismatch'] = int(sum(r.record_id not in ids.get(r.source, set()) for r in m.itertuples()))
D['dup_membership'] = int(m.record_id.duplicated().sum())
D['fused_rows'] = len(f); D['fused_ids_eq_clusters'] = set(f._id) == set(m.cluster_id)
sz = m.cluster_id.value_counts(); ns = m.groupby('cluster_id').source.nunique()
D['clusters'] = len(sz); D['singleton_share'] = round(float((sz == 1).mean()), 4); D['max_cluster'] = int(sz.max())
D['sources_per_cluster'] = {int(k): int(v) for k, v in ns.value_counts().sort_index().items()}
D['within_source_dup_clusters'] = int((m.groupby('cluster_id').source.apply(lambda s: s.duplicated().any())).sum())
D['correspondences'] = len(c)
cs = set(zip(b.id1, b.id2)) | set(zip(b.id2, b.id1))
D['corr_in_candidates'] = round(sum((x, y) in cs for x, y in zip(c.id1, c.id2)) / max(1, len(c)), 4)
D['candidates'] = len(b)
cl = dict(zip(m.record_id, m.cluster_id))
D['corr_same_cluster'] = round(sum(cl[x] == cl[y] for x, y in zip(c.id1, c.id2)) / max(1, len(c)), 4)
# expected correspondences from clusters
exp = sum(1 for _, g in m.groupby('cluster_id') for x, y in itertools.combinations(g.source, 2) if x != y)
D['corr_complete_wrt_clusters'] = exp == len(c)
dens = {}; invalid = {}
for col, spec in sch.items():
    if col == 'id': continue
    v = f[col].str.strip(); nn = v[v != '']; dens[col] = round(len(nn) / len(f), 3)
    bad = 0
    for x in nn:
        if spec.get('type') == 'number':
            try:
                y = float(x); bad += (('minimum' in spec and y < spec['minimum']) or ('maximum' in spec and y > spec['maximum']))
            except: bad += 1
        else:
            if 'enum' in spec and x not in spec['enum']: bad += 1
            if 'pattern' in spec and not re.fullmatch(spec['pattern'], x): bad += 1
            if 'maxLength' in spec and len(x) > spec['maxLength']: bad += 1
    invalid[col] = bad
D['density'] = dens; D['schema_invalid'] = {k: v for k, v in invalid.items() if v}
D['required_missing'] = {k: int((f[k].str.strip() == '').sum()) for k in ['product_type', 'brand', 'title']}
# provenance trace: every categorical fused value must equal (token-set) some member raw value or be a documented repair
a = pd.concat([d.assign(source=s) for s, d in srcs.items()])
out = {'D': D}
json.dump(D, sys.stdout, indent=0); print()
with open('work/diagnostics.jsonl', 'a') as fh:
    fh.write(json.dumps(dict(ts=datetime.datetime.now().isoformat(timespec='seconds'), revision=rev,
                             inputs=['submission/membership.csv', 'submission/fused.csv', 'submission/correspondences.csv', 'submission/blocking/candidates.csv'], **D)) + '\n')
