"""Label-free diagnostics over saved submission artifacts (NOT accuracy)."""
import os, re, json, time, pandas as pd, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
rev = os.environ.get('REVISION', 'current')
src_ids = {}
for s in ['crossref', 'dblp', 'open_alex']:
    src_ids[s] = set(pd.read_json(f'task/input/data/{s}.jsonl', lines=True, dtype=False).id)
M = pd.read_csv('submission/membership.csv'); F = pd.read_csv('submission/fused.csv', dtype=str)
Co = pd.read_csv('submission/correspondences.csv'); B = pd.read_csv('submission/blocking/candidates.csv')
schema = json.load(open('task/input/schemamatching/target_schema.json'))['properties']
allids = set().union(*src_ids.values())
bset = set(zip(B.id1, B.id2)) | set(zip(B.id2, B.id1))
sz = M.groupby('cluster_id').size()
nsrc = M.groupby('cluster_id').source.nunique()
d = dict(ts=time.strftime('%FT%T'), revision=rev, inputs='submission/*.csv',
    coverage={s: round(len(set(M.record_id[M.source == s]) & ids) / len(ids), 4) for s, ids in src_ids.items()},
    unresolved_ids=int((~M.record_id.isin(allids)).sum()) + int((~Co.id1.isin(allids) | ~Co.id2.isin(allids)).sum()),
    dup_membership=int(M.record_id.duplicated().sum()),
    corr_in_candidates=round(float(np.mean([(a, b) in bset for a, b in zip(Co.id1, Co.id2)])), 4),
    n_corr=len(Co), n_candidates=len(B), n_clusters=len(sz),
    cluster_size=sz.value_counts().sort_index().to_dict(), singleton_share=round(float((sz == 1).mean()), 4),
    multi_source_share=round(float((nsrc > 1).mean()), 4),
    within_source_multiplicity=int((M.groupby(['cluster_id', 'source']).size() > 1).sum()),
    fused_ids_match_membership=set(F._id) == set(M.cluster_id), fused_dup_ids=int(F._id.duplicated().sum()),
    density={c: round(float(F[c].notna().mean()), 4) for c in F.columns})
viol = {}
for a, sp in schema.items():
    if a not in F: continue
    v = F[a].dropna()
    bad = 0
    if 'enum' in sp: bad += int((~v.isin(sp['enum'])).sum())
    if 'pattern' in sp: bad += int((~v.str.match(sp['pattern'])).sum())
    if sp.get('type') == 'integer':
        x = pd.to_numeric(v, errors='coerce'); bad += int(x.isna().sum())
        if 'minimum' in sp: bad += int((x < sp['minimum']).sum())
        if 'maximum' in sp: bad += int((x > sp['maximum']).sum())
    if sp.get('type') == 'array':
        bad += int(v.map(lambda s: not (isinstance(json.loads(s), list) and len(json.loads(s)) >= 1)).sum())
    viol[a] = bad
fp = pd.to_numeric(F.first_page, errors='coerce'); lp = pd.to_numeric(F.last_page, errors='coerce')
viol['last_lt_first'] = int((lp < fp).sum())
d['schema_violations'] = viol
P = pd.read_csv('work/state/s6_provenance.csv', dtype=str)
d['disagreement_rate'] = {a: round(float(P[a].dropna().map(lambda s: int(s.split('/')[-1]) > 1 if '/' in s else False).mean()), 4)
                          for a in P.columns if a != '_id'}
open('work/diagnostics.jsonl', 'a').write(json.dumps(d) + '\n')
print(json.dumps(d, indent=1))
