"""Stage 6: fusion + export of all submission files from the final clustering."""
import re, json, math, itertools, os
from collections import Counter
import numpy as np, pandas as pd

d = pd.read_pickle('state/s4_clustered.pkl')
sc = pd.read_pickle('state/s4_scored.pkl')
cand = sc[['id1', 'id2', 'score']]
os.makedirs('../submission/blocking', exist_ok=True)

def s(x):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else str(x).strip()

# ---- cluster ids ----
seed = d.loc[d.groupby('cluster_key').apply(lambda g: g.index.min())]
cid_of_key = {k: f"{d.loc[k, 'source']}__{d.loc[k, 'id']}" for k in d['cluster_key'].unique()}
d['cluster_id'] = d['cluster_key'].map(cid_of_key)

SRC_RANK = {'dataset_1': 0, 'dataset_2': 1, 'dataset_3': 2, 'dataset_4': 3}

def vote(values, rank=None, key=None):
    """Majority vote over non-empty values; tie-break: earliest source rank, then lexical."""
    vals = [(v, r) for v, r in zip(values, rank or [0] * len(values)) if s(v) != '']
    if not vals: return ''
    k = key or (lambda v: v)
    cnt = Counter(k(v) for v, _ in vals)
    best = max(cnt.values())
    winners = {kk for kk, c in cnt.items() if c == best}
    cands = sorted([(r, str(v)) for v, r in vals if k(v) in winners])
    return [v for v, r in vals if (r, str(v)) == cands[0]][0]

def vote_num(values, rank, rel=0.02):
    vals = [(float(v), r) for v, r in zip(values, rank) if v == v and v is not None and s(v) != '']
    if not vals: return np.nan
    # cluster numerically-equivalent values (within rel tolerance)
    groups = []
    for v, r in sorted(vals, key=lambda x: x[1]):
        for g in groups:
            if abs(g[0][0] - v) <= rel * max(abs(g[0][0]), abs(v)): g.append((v, r)); break
        else: groups.append([(v, r)])
    groups.sort(key=lambda g: (-len(g), min(r for _, r in g)))
    g = groups[0]
    return Counter(v for v, _ in g).most_common(1)[0][0]

def normkey(v):
    return re.sub(r'[^a-z0-9]', '', s(v).lower())

def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

APPLIES = {'chipset_name': {'GPU'}, 'vram_gb': {'GPU'}, 'storage_gb': {'SSD', 'HDD', 'USB_STICK'},
           'read_speed_mb_s': {'SSD', 'HDD', 'USB_STICK'}, 'write_speed_mb_s': {'SSD', 'HDD', 'USB_STICK'},
           'storage_connection_type': {'SSD', 'HDD', 'USB_STICK'}, 'memory_type': {'GPU', 'SSD'}}
SCHEMA = json.load(open('../task/input/schemamatching/target_schema.json'))
TARGET = [c for c in SCHEMA['properties'] if c != 'id']
MPN_RE = re.compile(SCHEMA['properties']['model_number']['pattern'])

BUS_SPEC = lambda v: (len(v), v)
fused, prov = [], []
for cid, g in d.groupby('cluster_id', sort=True):
    g = g.sort_values('source', key=lambda x: x.map(SRC_RANK))
    rk = [SRC_RANK[x] for x in g['source']]
    toks = [set(t) for t in g['tok']]
    # representative record = medoid by title token overlap (tie: earliest source)
    cent = [np.mean([jacc(a, b) for b in toks]) for a in toks]
    rep = g.iloc[int(np.argmax(cent))]
    row = {'_id': cid}
    ptype = vote(list(g['product_type']), rk)
    row['product_type'] = ptype
    bvals = [b for b in g['n_brand'] if b]
    if ptype == 'GPU' and any(b not in ('NVIDIA', 'AMD') for b in bvals):
        bvals_r = [(b, r) for b, r in zip(g['n_brand'], rk) if b and b not in ('NVIDIA', 'AMD')]
    else:
        bvals_r = [(b, r) for b, r in zip(g['n_brand'], rk) if b]
    row['brand'] = vote([b for b, _ in bvals_r], [r for _, r in bvals_r], key=str.lower) if bvals_r else ''
    row['title'] = rep['n_title']
    row['description'] = rep['n_desc'][:10000]
    row['price'] = rep['n_price']
    row['priceCurrency'] = rep['n_cur'] if rep['n_price'] == rep['n_price'] else ''
    row['url'] = s(rep['url'])
    row['model'] = vote(list(g['model']), rk, key=normkey)
    mpns = [s(v) for v in g['model_number']]
    row['model_number'] = vote(mpns, rk, key=normkey)
    row['chipset_name'] = ''
    if ptype == 'GPU':
        # vote on normalized chip key, output most common raw spelling of the winner
        keys = list(g['n_chip'])
        kwin = vote(keys, rk)
        raws = [(s(c), r) for c, k, r in zip(g['chipset_name'], keys, rk) if k == kwin and s(c)]
        row['chipset_name'] = vote([c for c, _ in raws], [r for _, r in raws]) if raws else ''
        row['vram_gb'] = vote_num(list(g['n_vram_gb']), rk, rel=0.0)
    else:
        row['vram_gb'] = np.nan
    if ptype in APPLIES['storage_gb']:
        row['storage_gb'] = vote_num(list(g['n_storage_gb']), rk)
        row['read_speed_mb_s'] = vote_num(list(g['n_read_speed_mb_s']), rk)
        row['write_speed_mb_s'] = vote_num(list(g['n_write_speed_mb_s']), rk)
        row['storage_connection_type'] = vote(list(g['n_conn']), rk)
    else:
        row['storage_gb'] = row['read_speed_mb_s'] = row['write_speed_mb_s'] = np.nan
        row['storage_connection_type'] = ''
    row['bus_type'] = vote(list(g['n_bus']), rk)
    row['interface_type'] = vote(list(g['n_iface']), rk)
    for c in ['width_mm', 'length_mm', 'height_mm', 'weight_g']:
        row[c] = vote_num(list(g['n_' + c]), rk)
    mems = [m if m else '' for m in g['n_mem']]
    row['memory_type'] = vote(mems, rk) if ptype in APPLIES['memory_type'] else ''
    row['color'] = vote(list(g['n_color']), rk, key=str.lower)
    row['form_factor'] = vote(list(g['n_ff']), rk)
    fused.append(row)
    prov.append(dict(_id=cid, members=list(g['id']), rep_record=rep['id']))

F = pd.DataFrame(fused)
# schema range checks: out-of-range numerics become missing (honest missingness)
for c in ['vram_gb', 'storage_gb', 'read_speed_mb_s', 'write_speed_mb_s', 'width_mm', 'length_mm', 'height_mm', 'weight_g', 'price']:
    p = SCHEMA['properties'][c]
    lo, hi = p.get('minimum', -np.inf), p.get('maximum', np.inf)
    F.loc[(F[c] < lo) | (F[c] > hi), c] = np.nan
F.loc[~F['model_number'].map(lambda v: v == '' or bool(MPN_RE.match(v))), 'model_number'] = ''
F.loc[~F['priceCurrency'].map(lambda v: v == '' or bool(re.fullmatch(r'[A-Z]{3}', v))), 'priceCurrency'] = ''
for c in ['brand', 'model', 'chipset_name', 'color', 'form_factor', 'bus_type', 'interface_type', 'memory_type', 'storage_connection_type']:
    ml = SCHEMA['properties'][c].get('maxLength')
    if ml: F[c] = F[c].map(lambda v: v[:ml])
F = F[['_id'] + TARGET]
F.to_csv('../submission/fused.csv', index=False)
json.dump(prov, open('state/fusion_provenance.json', 'w'))

M = d[['id', 'source', 'cluster_id']].rename(columns={'id': 'record_id'})
M.to_csv('../submission/membership.csv', index=False)

# correspondences: every cross-source pair inside a cluster
pscore = {(a, b): v for a, b, v in zip(cand.id1, cand.id2, cand.score)}
corr = []
for cid, g in d.groupby('cluster_id'):
    for (i1, s1), (i2, s2) in itertools.combinations(zip(g['id'], g['source']), 2):
        if s1 == s2: continue
        v = pscore.get((i1, i2), pscore.get((i2, i1), np.nan))
        corr.append((i1, i2, v))
C = pd.DataFrame(corr, columns=['id1', 'id2', 'score'])
C['direct_candidate'] = C['score'].notna()
# score in [0,1]: logistic squashing of the rule score; transitive (non-candidate) pairs get 0.5
C['score'] = C['score'].map(lambda v: 0.5 if v != v else round(1 / (1 + math.exp(-6 * (v - 0.3))), 4))
C[['id1', 'id2', 'score']].to_csv('../submission/correspondences.csv', index=False)
C.to_csv('state/correspondences_detail.csv', index=False)

# blocking candidates: top-K/code-key pairs plus the transitive intra-cluster pairs that clustering compared through members
B = pd.concat([cand[['id1', 'id2']], C[['id1', 'id2']]]).drop_duplicates()
B.to_csv('../submission/blocking/candidates.csv', index=False)
pd.read_csv('state/sm_mapping.csv').to_csv('../submission/sm_mapping.csv', index=False)
print('fused', F.shape, 'membership', M.shape, 'corr', len(C), 'transitive-only', int((~C.direct_candidate).sum()), 'cands', len(B))
