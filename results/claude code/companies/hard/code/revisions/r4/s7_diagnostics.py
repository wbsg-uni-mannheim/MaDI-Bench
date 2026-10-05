"""Label-free diagnostic panel computed from the saved submission files (not correctness scores)."""
import os, sys, json, time, re
import pandas as pd, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = f'{ROOT}/submission'; D = f'{ROOT}/task/input/data'
def main(rev='current'):
    src = {s: pd.read_csv(f'{D}/{s}.csv', dtype=str) for s in ['dbpedia', 'forbes', 'fullcontact']}
    ids = {s: set(df.id) for s, df in src.items()}
    M = pd.read_csv(f'{SUB}/membership.csv', dtype=str); F = pd.read_csv(f'{SUB}/fused.csv', dtype=str)
    C = pd.read_csv(f'{SUB}/correspondences.csv', dtype={'id1': str, 'id2': str}); B = pd.read_csv(f'{SUB}/blocking/candidates.csv', dtype=str)
    tax_c = pd.read_csv(f'{ROOT}/task/input/schemamatching/CLDR_Country_Taxonomy.csv'); tax_i = pd.read_csv(f'{ROOT}/task/input/schemamatching/GICS_Industry_Taxonomy.csv')
    d = {'ts': time.time(), 'stage': 'panel', 'revision': rev, 'inputs': 'submission/*.csv'}
    allids = set().union(*ids.values())
    d['membership_rows'] = len(M); d['membership_unresolved_ids'] = int((~M.record_id.isin(allids)).sum())
    d['source_coverage'] = {s: round(len(ids[s] & set(M[M.source == s].record_id)) / len(ids[s]), 4) for s in ids}
    d['dup_membership'] = int(M.record_id.duplicated().sum())
    d['fused_ids_match_membership'] = set(F._id) == set(M.cluster_id)
    sz = M.groupby('cluster_id').size(); ns = M.groupby('cluster_id').source.nunique()
    d['clusters'] = len(sz); d['singleton_share'] = round(float((sz == 1).mean()), 4); d['max_cluster'] = int(sz.max())
    d['sources_per_cluster'] = {int(k): int(v) for k, v in ns.value_counts().items()}
    within = M.groupby(['cluster_id', 'source']).size(); d['clusters_with_within_source_multiplicity'] = int((within > 1).groupby(level=0).any().sum())
    bset = set(map(tuple, np.sort(B[['id1', 'id2']].values, axis=1)))
    cp = list(map(tuple, np.sort(C[['id1', 'id2']].values, axis=1)))
    d['correspondences'] = len(C); d['corr_in_candidates'] = round(sum(p in bset for p in cp) / max(len(cp), 1), 4)
    cl = dict(zip(M.record_id, M.cluster_id)); sm = dict(zip(M.record_id, M.source))
    d['corr_same_cluster'] = round(float(np.mean([cl[a] == cl[b] for a, b in cp])), 4) if cp else None
    d['corr_same_source'] = int(sum(sm[a] == sm[b] for a, b in cp))
    d['candidates'] = len(B)
    dens = {}
    for c in F.columns:
        v = F[c].fillna('').astype(str).str.strip(); dens[c] = round(float((v != '').mean()), 4)
    d['density'] = dens
    val = {}
    val['country_in_taxonomy'] = round(float(F.country.dropna().isin(set(tax_c['Country Name'])).mean()), 4)
    val['industry_in_taxonomy'] = round(float(F.industry.dropna().isin(set(tax_i['Industry Name'])).mean()), 4)
    fd = F.founded.dropna(); val['founded_pattern'] = round(float(fd.str.match(r'^\d{4}-\d{2}-\d{2}$').mean()), 4)
    val['founded_in_range'] = round(float(fd.between('1700-01-01', '2016-12-31').mean()), 4)
    for c, mx in [('assets', 4e12), ('revenue', 1e12)]:
        x = pd.to_numeric(F[c], errors='coerce').dropna(); val[f'{c}_int_in_range'] = round(float(((x >= 0) & (x <= mx) & (x == x.round())).mean()), 4)
    kp = F.keypeople.dropna(); val['keypeople_json_list'] = round(float(kp.map(lambda s: isinstance(json.loads(s), list)).mean()), 4)
    val['name_len_ok'] = round(float(F.name.dropna().str.len().between(1, 200).mean()), 4)
    d['validity'] = val
    print(json.dumps(d, indent=1))
    with open(f'{ROOT}/work/diagnostics.jsonl', 'a') as fh: fh.write(json.dumps(d) + '\n')
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'current')
