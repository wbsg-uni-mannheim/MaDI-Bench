"""Stage 6: attribute-wise fusion of the final clusters (work/state/membership.csv) from normalized member records."""
import os, re, json, collections
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
schema = json.load(open('task/input/schemamatching/target_schema.json'))
ATTRS = [a for a in schema['properties'] if a != 'id']
ENUM_PT = set(schema['properties']['product_type']['enum'])
d = pd.read_pickle('work/state/norm.pkl')
mb = pd.read_csv('work/state/membership.csv')
assert (mb.record_id.values == d.id.values).all()
d['cid'] = mb.cluster_id.values
tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True).fit(d.title_n)
X = tf.transform(d.title_n)

# global frequency of raw spellings, used as a deterministic tie-break toward the most common spelling
def ckey(v):
    return re.sub(r'[^a-z0-9]', '', str(v).lower())
USB = [(r'usb3\.?0|usb3\.?1gen1|usb3\.?2gen1|usb31gen1|usb32gen1', 'usb3gen1'), (r'usb3\.?1gen2|usb3\.?2gen2(?!x)', 'usb3gen2')]
def bus_key(v):
    k = ckey(v)
    for pat, rep in USB: k = re.sub(pat, rep, k)
    return k
spell_freq = {c: collections.Counter(d[c].dropna()) for c in list(ATTRS) + ['product_type_c'] if c in d.columns and d[c].dtype == object}

def clean_spaces(v, col):
    # undo injected single spaces inside a word ('Bl ack') when the de-spaced spelling is attested elsewhere in the column
    if not isinstance(v, str) or ' ' not in v: return v
    cand = v.replace(' ', '')
    for w, _ in spell_freq[col].most_common():
        if w.replace(' ', '') == cand and w != v and spell_freq[col][w] > spell_freq[col][v]:
            return w
    return v

def vote_str(vals, col, key=ckey):
    vals = [clean_spaces(v, col) for v in vals if isinstance(v, str) and v.strip()]
    if not vals: return None, 0, 0
    groups = collections.defaultdict(list)
    for v in vals: groups[key(v)].append(v)
    best = sorted(groups.items(), key=lambda kv: (-len(kv[1]), -sum(spell_freq[col].get(x, 0) for x in kv[1]), kv[0]))[0]
    forms = collections.Counter(best[1])
    rep = sorted(forms, key=lambda f: (-forms[f], -spell_freq[col].get(f, 0), f))[0]
    return rep, len(best[1]), len(groups)

def vote_num(vals, extra=(), tol=0.02):
    vals = [float(v) for v in vals if pd.notna(v)]
    ext = [float(v) for v in extra if pd.notna(v)]
    if not vals and not ext: return None, 0, 0
    pool = vals + ext
    def support(x):
        return sum(1.0 for v in vals if abs(v - x) <= tol * max(abs(x), 1e-9)) + 0.5 * sum(1.0 for v in ext if abs(v - x) <= tol * max(abs(x), 1e-9))
    cands = sorted(set(vals) if vals else set(ext))
    best = sorted(cands, key=lambda x: (-support(x), abs(x - float(np.median(pool))), x))[0]
    return best, support(best), len({round(v, 3) for v in vals})

NUMERIC = ['price', 'vram_gb', 'storage_gb', 'read_speed_mb_s', 'write_speed_mb_s', 'width_mm', 'length_mm', 'height_mm', 'weight_g']
rows, prov = [], []
for cid, g in d.groupby('cid', sort=True):
    idx = g.index.values
    # medoid record (most central title) supplies offer-level text fields
    if len(idx) > 1:
        S = (X[idx] @ X[idx].T).toarray(); med = idx[int(np.argmax(S.sum(1)))]
    else:
        med = idx[0]
    out = {'_id': cid}
    for a in ATTRS:
        if a in ('title', 'description', 'price', 'priceCurrency', 'url'):
            v = d.at[med, a]
            if a == 'priceCurrency' and isinstance(v, str):
                v = re.sub(r'\s+', '', v).upper()
                v = v if re.fullmatch(r'[A-Z]{3}', v) else None
            out[a] = v if (not isinstance(v, float) or pd.notna(v)) else None
            prov.append((cid, a, 'medoid:' + d.at[med, 'id']))
            continue
        if a == 'product_type':
            v, n, k = vote_str(g.product_type_c.tolist(), 'product_type_c', key=lambda x: x)
        elif a == 'brand':
            v, n, k = vote_str(g.brand.tolist(), 'brand', key=lambda x: str(x).strip().lower())
        elif a == 'bus_type':
            v, n, k = vote_str(g[a].tolist(), a, key=bus_key)
        elif a == 'storage_gb':
            v, n, k = vote_num(g.storage_gb.tolist(), extra=g.cap_key.tolist())
        elif a in NUMERIC:
            v, n, k = vote_num(g[a].tolist())
        else:
            v, n, k = vote_str(g[a].tolist(), a)
        out[a] = v
        prov.append((cid, a, f'vote support={n} distinct={k}'))
    # schema-validity guards
    if out['model_number'] is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._\-/()]*', str(out['model_number'])):
        alt = [m for m in g.model_number.dropna() if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._\-/()]*', m)]
        out['model_number'] = vote_str(alt, 'model_number')[0] if alt else None
    for a, lo, hi in [('vram_gb', 1, 128), ('storage_gb', 1, 100000), ('read_speed_mb_s', 1, 20000), ('write_speed_mb_s', 1, 20000),
                      ('width_mm', 0.1, 1000), ('length_mm', 0.1, 1000), ('height_mm', 0.1, 300), ('weight_g', 0.5, 10000)]:
        if out[a] is not None and not (lo <= out[a] <= hi): out[a] = None
    out['member_count'] = len(idx)
    rows.append(out)
F = pd.DataFrame(rows)
F.drop(columns=['member_count']).to_csv('work/state/fused.csv', index=False)
pd.DataFrame(prov, columns=['cluster_id', 'attribute', 'rule']).to_csv('work/state/fusion_provenance.csv', index=False)
dens = F[ATTRS].notna().mean().round(3).to_dict()
stats = {'fused_rows': len(F), 'density': dens, 'product_type_in_enum': float(F.product_type.isin(ENUM_PT).mean())}
print(json.dumps(stats)); json.dump(stats, open('work/state/fusion_stats.json', 'w'))
