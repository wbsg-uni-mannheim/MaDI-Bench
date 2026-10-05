"""Label-free diagnostics over the saved submission artifacts (NOT accuracy scores)."""
import json, sys, datetime, itertools
import pandas as pd, numpy as np

rev = sys.argv[1] if len(sys.argv) > 1 else 'current'
S = '../submission/'
M = pd.read_csv(S + 'membership.csv', dtype=str)
F = pd.read_csv(S + 'fused.csv', dtype={'_id': str})
C = pd.read_csv(S + 'correspondences.csv', dtype={'id1': str, 'id2': str})
B = pd.read_csv(S + 'blocking/candidates.csv', dtype=str)
d = pd.read_pickle('state/s4_clustered.pkl')
sc = pd.read_pickle('state/s4_scored.pkl')

out = {'revision': rev, 'time': datetime.datetime.now().isoformat(timespec='seconds'),
       'inputs': ['submission/membership.csv', 'submission/fused.csv', 'submission/correspondences.csv', 'submission/blocking/candidates.csv']}
src_ids = set(zip(d['source'], d['id']))
out['records_total'] = len(src_ids)
out['membership_rows'] = len(M)
out['membership_unresolved'] = int(sum((s, i) not in src_ids for s, i in zip(M.source, M.record_id)))
out['records_missing_from_membership'] = len(src_ids - set(zip(M.source, M.record_id)))
out['clusters_without_fused_row'] = len(set(M.cluster_id) - set(F['_id']))
out['fused_rows_without_members'] = len(set(F['_id']) - set(M.cluster_id))
sz = M.groupby('cluster_id').size()
out['n_entities'] = int(len(sz))
out['cluster_size_dist'] = {int(k): int(v) for k, v in sz.value_counts().sort_index().items()}
out['singleton_share'] = round(float((sz == 1).mean()), 4)
out['max_same_source_in_cluster'] = int(M.groupby(['cluster_id', 'source']).size().max())
nsrc = M.groupby('cluster_id').source.nunique()
out['sources_per_cluster'] = {int(k): int(v) for k, v in nsrc.value_counts().sort_index().items()}
# ds1/ds2 pairing (structural prior: identical per-type counts)
has = M.groupby('cluster_id').source.agg(set)
out['clusters_with_ds1_and_ds2'] = int(has.map(lambda x: {'dataset_1', 'dataset_2'} <= x).sum())
Bs = set(map(tuple, B[['id1', 'id2']].values)) | set(map(tuple, B[['id2', 'id1']].values))
out['candidates'] = len(B)
out['correspondences'] = len(C)
out['corr_in_candidates'] = round(float(np.mean([(a, b) in Bs for a, b in zip(C.id1, C.id2)])), 4)
sc2 = sc.set_index(['id1', 'id2'])
direct = [(a, b) for a, b in zip(C.id1, C.id2) if (a, b) in sc2.index or (b, a) in sc2.index]
out['corr_direct_scored_share'] = round(len(direct) / max(1, len(C)), 4)
# contradictions inside accepted correspondences (from the scored features)
feats = []
for a, b in direct:
    r = sc2.loc[(a, b)] if (a, b) in sc2.index else sc2.loc[(b, a)]
    feats.append(r)
FF = pd.DataFrame(feats)
for f in ['cap', 'chip', 'vram', 'brand', 'mem', 'mtok', 'rpm', 'iface', 'ffk', 'sasg', 'mpn']:
    out[f'corr_conflict_{f}'] = int((FF[f] < 0).sum())
out['corr_conflict_excl'] = int((FF['excl'] > 0).sum())
# blocking stats
tot = 0
for t, g in d.groupby('product_type'):
    cnt = g.source.value_counts().values
    tot += sum(a * b for a, b in itertools.combinations(cnt, 2))
out['typed_cross_source_pairs'] = int(tot)
out['reduction_ratio_vs_all_cross_source'] = round(1 - len(B) / (sum(a * b for a, b in itertools.combinations(d.source.value_counts().values, 2))), 4)
deg = pd.concat([B.id1, B.id2]).value_counts()
out['records_with_zero_candidates'] = int(len(set(d.id) - set(deg.index)))
# schema / density
out['fused_density'] = {c: round(float(F[c].notna().mean()), 3) for c in F.columns if c != '_id'}
sch = json.load(open('../task/input/schemamatching/target_schema.json'))['properties']
viol = {}
for c, p in sch.items():
    if c not in F.columns: continue
    v = F[c].dropna()
    if 'enum' in p: viol[c] = int((~v.isin(p['enum'])).sum())
    if 'minimum' in p: viol[c] = int(((v < p['minimum']) | (v > p.get('maximum', np.inf))).sum())
out['schema_violations'] = {k: v for k, v in viol.items() if v}
for c, ts in [('chipset_name', {'GPU'}), ('vram_gb', {'GPU'}), ('storage_gb', {'SSD', 'HDD', 'USB_STICK'})]:
    out[f'applicability_violations_{c}'] = int((F[c].notna() & ~F.product_type.isin(ts)).sum())
# fused rows belong to their own members: rep title must come from a member
prov = json.load(open('state/fusion_provenance.json'))
mem = M.groupby('cluster_id').record_id.agg(set)
out['fused_rep_outside_cluster'] = int(sum(str(p['rep_record']) not in mem[p['_id']] for p in prov))
# disagreement rate among multi-member clusters for key attributes (normalized)
dis = {}
for col in ['n_storage_gb', 'n_vram_gb', 'n_chip', 'n_brand_fam', 'n_mem', 'n_bus', 'n_ff']:
    g = d.groupby('cluster_key')[col].agg(lambda x: len(set(v for v in x if v == v and v != '')))
    multi = d.groupby('cluster_key').size() > 1
    dis[col] = round(float((g[multi] > 1).mean()), 4)
out['cluster_value_disagreement_rate'] = dis
print(json.dumps(out, indent=1))
with open('diagnostics.jsonl', 'a') as f:
    f.write(json.dumps(out) + '\n')
