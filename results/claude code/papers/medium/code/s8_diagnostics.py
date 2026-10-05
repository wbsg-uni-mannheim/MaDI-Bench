"""Label-free diagnostics panel computed from the saved submission files."""
import pandas as pd, json, re, time, sys
REV = sys.argv[1] if len(sys.argv) > 1 else 'current'
S = 'submission/'
src = {s: pd.read_csv(f'task/input/data/{s}.csv', dtype=str, usecols=['id']).id for s in ['crossref', 'dblp', 'open_alex']}
allids = {i: s for s, v in src.items() for i in v}
fz = pd.read_csv(S + 'fused.csv', dtype=str, keep_default_na=False)
mem = pd.read_csv(S + 'membership.csv', dtype=str)
cor = pd.read_csv(S + 'correspondences.csv', dtype={'id1': str, 'id2': str})
cand = pd.read_csv(S + 'blocking/candidates.csv', dtype=str)
schema = json.load(open('task/input/schemamatching/target_schema.json'))['properties']
d = {'revision': REV, 'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
     'inputs': [S + x for x in ['fused.csv', 'membership.csv', 'correspondences.csv', 'blocking/candidates.csv']]}
d['unresolved_membership_ids'] = int((~mem.record_id.isin(allids)).sum())
d['source_label_mismatch'] = int((mem.record_id.map(allids) != mem.source).sum())
d['records_covered'] = {s: float(v.isin(mem.record_id).mean()) for s, v in src.items()}
d['dup_membership'] = int(mem.record_id.duplicated().sum())
d['cluster_ids_not_in_fused'] = int((~mem.cluster_id.isin(fz._id)).sum())
d['fused_ids_not_in_membership'] = int((~fz._id.isin(mem.cluster_id)).sum())
key = set(zip(cand.id1, cand.id2)) | set(zip(cand.id2, cand.id1))
d['corr_in_candidates'] = float(pd.Series([(a, b) in key for a, b in zip(cor.id1, cor.id2)]).mean())
rc = dict(zip(mem.record_id, mem.cluster_id))
d['corr_same_cluster'] = float(pd.Series([rc[a] == rc[b] for a, b in zip(cor.id1, cor.id2)]).mean())
d['corr_same_source'] = int((cor.id1.map(allids) == cor.id2.map(allids)).sum())
g = mem.groupby('cluster_id')
d['n_clusters'] = int(len(fz)); d['cluster_size_dist'] = g.size().value_counts().sort_index().to_dict()
d['source_combo'] = g.source.agg(lambda s: '+'.join(sorted(s))).value_counts().to_dict()
d['max_same_source_in_cluster'] = int(mem.groupby(['cluster_id', 'source']).size().max())
d['singleton_share'] = float((g.size() == 1).mean())
d['n_candidates'] = int(len(cand)); d['n_correspondences'] = int(len(cor))
# schema validity / density
dens, inval = {}, {}
for c in fz.columns[1:]:
    v = fz[c].str.strip(); nn = v != ''
    dens[c] = round(float(nn.mean()), 4)
    p = schema[c]; bad = 0
    if 'enum' in p: bad = (~v[nn].isin(p['enum'])).sum()
    elif p.get('type') == 'integer':
        x = pd.to_numeric(v[nn], errors='coerce'); bad = (x.isna() | (x < p.get('minimum', -1e18)) | (x > p.get('maximum', 1e18))).sum()
    elif p.get('type') == 'array':
        l = v[nn].map(json.loads); bad = (l.map(len) == 0).sum() + l.map(lambda a: any(not x.strip() for x in a)).sum()
    elif 'pattern' in p:
        bad = (~v[nn].str.match(p['pattern']) | (v[nn].str.len() > p.get('maxLength', 1e9))).sum()
    elif p.get('type') == 'string':
        bad = (v[nn].str.len() > p.get('maxLength', 1e9)).sum()
    inval[c] = int(bad)
d['density'] = dens; d['invalid'] = inval
d['last_lt_first'] = int(((fz.first_page.str.isdigit()) & (fz.last_page.str.isdigit()) & (fz.last_page.str.isdigit() & fz.first_page.str.isdigit())).pipe(lambda m: (pd.to_numeric(fz.last_page[m]) < pd.to_numeric(fz.first_page[m])).sum()))
# traceability: fused value present among member normalized values
u = pd.read_pickle('work/state/s2_normalized.pkl').set_index('id')
smp = fz.sample(2000, random_state=0); miss = {}
colmap = {'type': 'type_n', 'title': 'title_n', 'publication_year': 'year_n', 'journal': 'journal_n', 'volume': 'volume_n', 'issue': 'issue_n',
          'first_page': 'first_page_n', 'last_page': 'last_page_n', 'referenced_works_count': 'ref_n', 'cited_by_count': 'cite_n'}
cl = g.record_id.apply(list).to_dict()
for c, n in colmap.items():
    miss[c] = int(sum(1 for _, r in smp.iterrows() if r[c] != '' and r[c] not in set(u.loc[cl[r._id], n])))
miss['authors'] = int(sum(1 for _, r in smp.iterrows() if r.authors != '' and json.loads(r.authors) not in list(u.loc[cl[r._id], 'authors_l'])))
d['untraceable_in_sample_2000'] = miss
json.dump(d, open('work/state/diagnostics_latest.json', 'w'), indent=1, default=int)
with open('work/diagnostics.jsonl', 'a') as f: f.write(json.dumps(d, default=int) + '\n')
print(json.dumps(d, indent=1, default=int))
