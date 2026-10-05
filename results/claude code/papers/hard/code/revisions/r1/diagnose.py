"""Label-free diagnostic panel computed from the saved submission artifacts (not a correctness score)."""
import pandas as pd, numpy as np, json, os, re, time, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = sys.argv[1] if len(sys.argv) > 1 else 'current'

def main():
    S = f'{BASE}/submission'
    fused = pd.read_csv(f'{S}/fused.csv', dtype=str, keep_default_na=False)
    mem = pd.read_csv(f'{S}/membership.csv', dtype=str, keep_default_na=False)
    corr = pd.read_csv(f'{S}/correspondences.csv', dtype=str, keep_default_na=False)
    cand = pd.read_csv(f'{S}/blocking/candidates.csv', dtype=str, keep_default_na=False)
    schema = json.load(open(f'{BASE}/task/input/schemamatching/target_schema.json'))['properties']
    src_ids = set()
    for s in ['crossref', 'dblp', 'open_alex']:
        src_ids |= set(pd.read_csv(f'{BASE}/task/input/data/{s}.csv', dtype=str, usecols=['id']).id)
    d = {}
    d['source_records'] = len(src_ids)
    d['membership_rows'] = len(mem)
    d['records_missing_from_membership'] = len(src_ids - set(mem.record_id))
    d['unknown_ids_in_membership'] = len(set(mem.record_id) - src_ids)
    d['duplicate_membership_rows'] = int(mem.record_id.duplicated().sum())
    d['fused_rows'] = len(fused)
    d['fused_ids_eq_cluster_ids'] = set(fused._id) == set(mem.cluster_id)
    d['unknown_ids_in_corr'] = len((set(corr.id1) | set(corr.id2)) - src_ids)
    cid = dict(zip(mem.record_id, mem.cluster_id))
    d['corr_pairs_split_across_clusters'] = int(sum(cid[a] != cid[b] for a, b in zip(corr.id1, corr.id2)))
    d['corr_same_source_pairs'] = int((corr.id1.str.split('-').str[0] == corr.id2.str.split('-').str[0]).sum())
    cs = set(zip(cand.id1, cand.id2)) | set(zip(cand.id2, cand.id1))
    d['corr_in_candidates_rate'] = round(float(np.mean([(a, b) in cs for a, b in zip(corr.id1, corr.id2)])), 4)
    d['candidate_pairs'] = len(cand)
    sz = mem.groupby('cluster_id').size()
    d['cluster_size_dist'] = {int(k): int(v) for k, v in sz.value_counts().sort_index().items()}
    d['sources_per_cluster'] = {int(k): int(v) for k, v in mem.groupby('cluster_id').source.nunique().value_counts().sort_index().items()}
    d['singleton_share'] = round(float((sz == 1).mean()), 4)
    d['clusters_with_same_source_multiples'] = int((mem.groupby(['cluster_id', 'source']).size() > 1).groupby(level=0).any().sum())
    # schema validity & density
    dens, bad = {}, {}
    for col in fused.columns[1:]:
        v = fused[col]
        nn = v != ''
        dens[col] = round(float(nn.mean()), 4)
        p = schema[col]
        b = 0
        if 'enum' in p:
            b = int((nn & ~v.isin(p['enum'])).sum())
        elif p['type'] == 'integer':
            num = pd.to_numeric(v[nn], errors='coerce')
            b = int(num.isna().sum() + ((num < p.get('minimum', -1e18)) | (num > p.get('maximum', 1e18))).sum())
        elif p['type'] == 'array':
            L = v[nn].map(json.loads)
            b = int((L.map(len) == 0).sum() + (L.map(len) > p.get('maxItems', 1e9)).sum())
        elif 'pattern' in p:
            b = int((nn & ~v.str.fullmatch(p['pattern'])).sum())
        if p.get('maxLength'):
            b += int((v.str.len() > p['maxLength']).sum())
        bad[col] = b
    d['density'] = dens
    d['schema_violations'] = bad
    fp, lp = pd.to_numeric(fused.first_page, errors='coerce'), pd.to_numeric(fused.last_page, errors='coerce')
    d['last_lt_first'] = int((lp < fp).sum())
    # traceability: fused title/volume/first_page must be supported by a member's (normalized) value
    n = pd.read_pickle(f'{BASE}/work/state/s2_normalized.pkl').set_index('id')
    members = mem.groupby('cluster_id').record_id.apply(list)
    samp = fused.sample(min(3000, len(fused)), random_state=0)
    miss = {'title': 0, 'volume': 0, 'first_page': 0, 'journal': 0}
    for _, r in samp.iterrows():
        M = n.loc[members[r._id]]
        if r.title and not any(r.title == t or r.title == t[:-1] for t in M.title_c):
            miss['title'] += 1
        if r.volume and r.volume not in set(M.volume_n):
            miss['volume'] += 1
        if r.first_page and r.first_page not in set(M.first_page_n):
            miss['first_page'] += 1
        if r.journal and r.journal not in set(M.journal_c):
            miss['journal'] += 1
    d['untraceable_in_sample_of_3000'] = miss
    # disagreement within multi-source clusters (normalized year / volume)
    nm = n.join(mem.set_index('record_id').cluster_id)
    multi = nm[nm.cluster_id.map(sz) > 1]
    for col in ['year_n', 'volume_n', 'first_page_n', 'type_n']:
        g = multi[multi[col].notna() & (multi[col].astype(str) != '')].groupby('cluster_id')[col].nunique()
        d[f'clusters_disagree_{col}'] = round(float((g > 1).mean()), 4)
    print(json.dumps(d, indent=1))
    with open(f'{BASE}/work/diagnostics.jsonl', 'a') as f:
        f.write(json.dumps({'stage': 'panel', 'revision': REV, 'input': 'submission/*',
                            'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), **d}) + '\n')

if __name__ == '__main__':
    main()
