"""Stage 6: attribute-wise fusion over final clusters, plus export of all submission files.
Every fused value comes from the cluster's own member records (raw value, or a documented
parse/normalization of it). Provenance per cell -> work/state/fusion_provenance.csv."""
import pandas as pd, numpy as np, os, re, json, itertools, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SUB = f'{ROOT}/submission'
from s2_normalize import parse_num
n = pd.read_pickle(f'{W}/state/normalized.pkl').set_index('id', drop=False)
cl = pd.read_csv(f'{W}/state/clusters.csv')
schema = json.load(open(f'{ROOT}/task/input/schemamatching/target_schema.json'))
PROPS = list(schema['properties'])
INSCOPE = ['GPU', 'SSD', 'HDD', 'USB_STICK']

def key_text(v):
    return re.sub(r'[^a-z0-9]+', '', str(v).lower().replace('®', '').replace('™', ''))

# global frequency of normalized forms -> tie-break (rare forms are usually OCR-corrupted)
GLOBAL = {c: Counter(key_text(v) for v in n[c] if v) for c in n.columns if n[c].dtype == object and c in PROPS}
RAWFREQ = {c: Counter(v for v in n[c] if v) for c in GLOBAL}

def vote_text(vals, col):
    """Majority over normalized keys; representative = most frequent raw spelling in the winning group."""
    vals = [v for v in vals if v and str(v).strip()]
    if not vals: return None, 'none'
    groups = defaultdict(list)
    for v in vals: groups[key_text(v)].append(v)
    groups.pop('', None)
    if not groups: return None, 'none'
    best = max(groups, key=lambda k: (len(groups[k]), GLOBAL[col][k], k))
    raw = max(groups[best], key=lambda v: (groups[best].count(v), RAWFREQ[col][v], v))
    how = 'unanimous' if len(groups) == 1 else f'vote {len(groups[best])}/{len(vals)}'
    return raw, how

def vote_num(vals, tol=0.02, lo=None):
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if lo is not None: vals = [v for v in vals if v >= lo]   # zeros/out-of-range = unknown
    if not vals: return None, 'none'
    best, bestc = None, -1
    for v in sorted(set(vals)):
        c = sum(1 for w in vals if abs(w - v) <= tol * max(abs(v), abs(w), 1e-9))
        if c > bestc: best, bestc = v, c
    how = 'unanimous' if len(set(vals)) == 1 else f'vote {bestc}/{len(vals)}'
    return best, how

def record_storage(r):
    """Per-record storage value: the source field unless it contradicts the title (unit slip -> title value)."""
    f = parse_num(r['storage_gb'], thousands_bias=True)
    t = r['n_cap_gb']
    if f is None or f <= 0: return (None if pd.isna(t) else t)
    if pd.isna(t) or t is None: return f
    if 0.9 <= f / t <= 1.1: return f
    return t

def record_vram(r):
    f = parse_num(r['vram_gb'])
    if f is not None and f > 128: f = f / 1024
    if f is not None and f >= 1: return f
    return None if pd.isna(r['n_vram_gb']) else r['n_vram_gb']

def medoid(ids):
    if len(ids) == 1: return ids[0]
    toks = {i: set(re.findall(r'[a-z0-9]+', n.loc[i, 'title'].lower())) for i in ids}
    def sim(a, b): return len(toks[a] & toks[b]) / max(1, len(toks[a] | toks[b]))
    return max(sorted(ids), key=lambda i: sum(sim(i, j) for j in ids if j != i))

rows, prov = [], []
for cid, g in cl.groupby('cluster', sort=True):
    ids = sorted(g.id)
    m = n.loc[ids]
    med = medoid(ids)
    out = {'_id': cid, 'id': cid}
    def put(col, val, how):
        out[col] = val; prov.append((cid, col, how, None if val is None else str(val)[:80]))
    # product type: canonical in-scope value by vote; out-of-scope kept as most common source spelling
    pts = [p for p in m.n_product_type if p]
    pt, how = vote_text(pts, 'product_type') if pts else (None, 'none')
    if pt and pt not in INSCOPE:
        raw = [v for v, p in zip(m.product_type, m.n_product_type) if p == pt and v]
        pt, how = vote_text(raw, 'product_type') if raw else (pt, how)
    put('product_type', pt, how)
    b, how = vote_text(list(m.brand), 'brand')
    if b is None:
        tb = [x for x in m.n_brand if x]
        if tb: b, how = Counter(tb).most_common(1)[0][0], 'title-detected brand'
    put('brand', b, how)
    put('title', n.loc[med, 'title'], 'medoid title')
    put('description', n.loc[med, 'description'] or None, 'medoid')
    pr = n.loc[med, 'n_price']
    put('price', None if pd.isna(pr) else pr, 'medoid record (paired with its currency)')
    put('priceCurrency', n.loc[med, 'n_currency'] if not pd.isna(pr) else None, 'medoid record')
    urls = [u for u in m.url if u]
    put('url', (n.loc[med, 'url'] or (urls[0] if urls else None)) or None, 'medoid or first url')
    for c in ['model', 'model_number', 'bus_type', 'interface_type', 'storage_connection_type', 'color', 'form_factor']:
        v, how = vote_text(list(m[c]), c)
        if c == 'model_number' and v: v = re.sub(r'^[^A-Za-z0-9]+', '', v) or None   # schema pattern: must start alnum
        put(c, v, how)
    is_gpu = pt == 'GPU'
    if is_gpu or pt not in INSCOPE:
        v, how = vote_text(list(m.chipset_name), 'chipset_name'); put('chipset_name', v, how)
        v, how = vote_num([record_vram(r) for _, r in m.iterrows()], lo=1); put('vram_gb', v, how)
    else:
        put('chipset_name', None, 'n/a for type'); put('vram_gb', None, 'n/a for type')
    if not is_gpu:
        v, how = vote_num([record_storage(r) for _, r in m.iterrows()], lo=1); put('storage_gb', v, how)
        for c in ['read_speed_mb_s', 'write_speed_mb_s']:
            v, how = vote_num(list(m['n_' + c]), lo=1); put(c, v, how)
    else:
        for c in ['storage_gb', 'read_speed_mb_s', 'write_speed_mb_s']: put(c, None, 'n/a for type')
    v, how = vote_text(list(m.memory_type), 'memory_type'); put('memory_type', v, how)
    for c in ['width_mm', 'length_mm', 'height_mm']:
        v, how = vote_num(list(m['n_' + c]), lo=0.1); put(c, v, how)
    # weights below the schema minimum (0.5 g) are kilogram values (e.g. 0,008 for an 8 g M.2 SSD) -> grams
    wts = [None if pd.isna(w) else (w * 1000 if 0 < w < 0.5 else w) for w in m['n_weight_g']]
    v, how = vote_num(wts, lo=0.5); put('weight_g', v, how)
    rows.append(out)

fused = pd.DataFrame(rows)[['_id'] + PROPS]
os.makedirs(SUB, exist_ok=True); os.makedirs(f'{SUB}/blocking', exist_ok=True)
fused.to_csv(f'{SUB}/fused.csv', index=False)
pd.DataFrame(prov, columns=['cluster', 'attribute', 'rule', 'value']).to_csv(f'{W}/state/fusion_provenance.csv', index=False)

# membership
cl.rename(columns={'id': 'record_id'})[['record_id', 'source', 'cluster']].rename(columns={'cluster': 'cluster_id'}) \
    .to_csv(f'{SUB}/membership.csv', index=False)

# correspondences: every cross-source pair inside a cluster; score = pairwise evidence score (capped to [0,1])
import s4_match as M
sc = pd.read_csv(f'{W}/state/scored.csv', keep_default_na=False)
known = {frozenset((a, b)): float(s) for a, b, s in zip(sc.id1, sc.id2, sc.score)}
corr = []
for cid, g in cl.groupby('cluster'):
    for a, b in itertools.combinations(sorted(g.id), 2):
        if n.loc[a, 'source'] == n.loc[b, 'source']: continue
        k = frozenset((a, b))
        s = known[k] if k in known else M.evidence(a, b)['score']
        corr.append((a, b, round(min(1.0, max(0.0, s)), 4)))
corr = pd.DataFrame(corr, columns=['id1', 'id2', 'score'])
corr.to_csv(f'{SUB}/correspondences.csv', index=False)

# blocking candidates: generated candidates + all within-cluster pairs the constrained clustering evaluated
cand = pd.read_csv(f'{W}/state/candidates.csv')[['id1', 'id2']]
extra = corr[['id1', 'id2']]
allc = pd.concat([cand, extra])
allc['k'] = [tuple(sorted(x)) for x in zip(allc.id1, allc.id2)]
allc = allc.drop_duplicates('k')
pd.DataFrame(allc.k.tolist(), columns=['id1', 'id2']).to_csv(f'{SUB}/blocking/candidates.csv', index=False)
print('fused', len(fused), 'corr', len(corr), 'cands', len(allc), 'closure-only cands', len(allc) - len(cand))
