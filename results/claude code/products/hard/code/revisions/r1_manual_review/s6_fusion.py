"""Stage 6: fusion. One row per cluster from s5 membership; every value comes from the cluster's own
member records (raw source values or documented transformations of them, e.g. unit/locale normalisation,
title-derived capacity/VRAM). Per-cell provenance is written to work/state/s6_provenance.csv."""
import pandas as pd, numpy as np, re, sys, json, difflib
from collections import Counter, defaultdict
sys.path.insert(0, 'work')
from normlib import *
a = pd.read_csv('work/state/s1_translated.csv', dtype=str, keep_default_na=False)
n = pd.read_csv('work/state/s2_features.csv', dtype=str, keep_default_na=False)
m = pd.read_csv('work/state/s5_membership.csv', dtype=str, keep_default_na=False)
d = a.merge(n, on=['id', 'source']).merge(m[['record_id', 'cluster_id']], left_on='id', right_on='record_id')
schema = json.load(open('task/input/schemamatching/target_schema.json'))
COLS = [c for c in schema['properties'] if c != 'id']

def toks(s): return set(re.findall(r'[a-z0-9]+', str(s).lower()))
d['text_toks'] = [toks(t + ' ' + de + ' ' + u) for t, de, u in zip(d.title, d.description, d.url)]

# ---------- global vocabularies (used to recognise corrupted categorical values) ----------
def ckey(v): return re.sub(r'[^a-z0-9]', '', str(v).lower().replace('®', '').replace('™', ''))
gcount = {c: Counter(ckey(v) for v in d[c] if v) for c in COLS}
graw = {c: Counter(v for v in d[c] if v) for c in COLS}

def clean_cat(c, v, cluster_toks):
    """A categorical value is trusted if its key occurs in >=2 records globally, or all of its
    alphanumeric tokens occur in the member titles/descriptions/urls."""
    if not v: return False
    if gcount[c][ckey(v)] >= 2: return True
    tk = toks(v)
    return bool(tk) and tk <= cluster_toks

def repair_cat(c, v):
    """Map an untrusted value to a close trusted global value with identical digits (typo repair)."""
    vocab = [k for k, cnt in gcount[c].items() if cnt >= 3]
    k = ckey(v)
    best = difflib.get_close_matches(k, vocab, n=1, cutoff=0.75)
    if best and re.sub(r'\D', '', best[0]) == re.sub(r'\D', '', k):
        # most common raw spelling of that key
        raws = [r for r in graw[c] if ckey(r) == best[0]]
        return max(raws, key=lambda r: (graw[c][r], r))
    return None

def tkey(v): return ' '.join(sorted(re.findall(r'[a-z0-9]+', str(v).lower().replace('®', ' ').replace('™', ' '))))
def nstr(v): return re.sub(r'[^a-z0-9]', '', str(v).lower())

def vote_cat(c, g, cluster_toks, prov):
    """Vote over trusted values grouped by token set (word order ignored, so token-shuffled copies count
    for the same value). Ties between groups: consensus (mean token overlap with the other values), then
    global frequency. Representation inside the winning group: the spelling that occurs verbatim
    (punctuation-insensitive) in a member title/description, else the most frequent spelling."""
    vals = [v for v in g[c] if v]
    good = [v for v in vals if clean_cat(c, v, cluster_toks)]
    how = 'vote'
    if not good:
        rep = [repair_cat(c, v) for v in vals]
        good = [r for r in rep if r]; how = 'typo_repair'
    if not good: return '', 'none' if not vals else 'rejected_corrupt'
    kc = Counter(tkey(v) for v in good)
    tsets = {k: set(k.split()) for k in kc}
    def consensus(k):
        o = [tsets[j] for j in kc for _ in range(kc[j]) if j != k]
        return np.mean([len(tsets[k] & t) / max(1, len(tsets[k] | t)) for t in o]) if o else 0
    k = max(kc, key=lambda k: (kc[k], round(consensus(k), 6), sum(gcount[c][ckey(v)] for v in good if tkey(v) == k), k))
    cand = [v for v in good if tkey(v) == k]
    texts = [nstr(t) + ' ' + nstr(de) for t, de in zip(g.title, g.description)]
    rc = Counter(cand)
    val = max(rc, key=lambda r: (any(nstr(r) and nstr(r) in t for t in texts), rc[r], graw[c][r], r))
    if not any(nstr(val) in t for t in texts) and gcount[c][ckey(val)] < 2 and c != 'storage_connection_type':
        # token-shuffled value: restore the word order observed in a member title/description
        words = val.split()
        for t in list(g.title) + list(g.description):
            tl = t.lower()
            pos = [tl.find(w.lower()) for w in words]
            if len(words) > 1 and all(x >= 0 for x in pos) and len(set(pos)) == len(pos):
                re_ordered = ' '.join(w for _, w in sorted(zip(pos, words)))
                if re_ordered != val: val = re_ordered; how += '+reordered'
                break
    return val, f'{how}:{kc[k]}/{len(vals)}'

OCR = str.maketrans({'O': '0', 'o': '0', 'B': '8', 'G': '6', 'S': '5', 's': '5', 'l': '1', 'I': '1', 'i': '1', 'Z': '2', 'z': '2'})
def num(v, ocr=False):
    x = parse_num(v)
    if x is None and ocr and v:
        x = parse_num(str(v).translate(OCR))
    return x

def vote_num(cands, tol=0.02):
    """cands: list of (value, weight). Cluster values within tol; return representative of heaviest cluster."""
    cands = [(v, w) for v, w in cands if v is not None and v > 0]
    if not cands: return None, 0
    best = None
    for v, _ in cands:
        sup = sum(w for u, w in cands if abs(u - v) / max(u, v) <= tol)
        if best is None or sup > best[1] or (sup == best[1] and v > best[0]): best = (v, sup)
    # representative: most frequent exact value among supporters
    sup_vals = Counter(u for u, w in cands if abs(u - best[0]) / max(u, best[0]) <= tol for _ in range(int(w * 2)))
    return max(sup_vals, key=lambda u: (sup_vals[u], u)), best[1]

BRAND_RAW_PREF = {}
for v, cnt in graw['brand'].items():
    k, how = brand_from_field(v)
    if k and how == 'exact': BRAND_RAW_PREF.setdefault(k, Counter())[v] += cnt
CANON_TYPES = {'GPU', 'SSD', 'HDD', 'USB_STICK'}

rows = []; prov = []
for cid, g in d.groupby('cluster_id', sort=True):
    ctoks = set().union(*g.text_toks)
    out = {'_id': cid}; P = {}
    # medoid record for free-text / offer-level fields (title, description, price, url)
    tt = [toks(t) for t in g.title]
    sc = [sum(len(x & y) / max(1, len(x | y)) for y in tt) for x in tt]
    med = g.iloc[int(np.argmax(sc))]
    for c in ['title', 'description', 'url']:
        out[c] = med[c]; P[c] = f'medoid:{med.id}'
    pr = num(med.price); out['price'] = pr if pr is not None else ''; P['price'] = f'medoid:{med.id}'
    cur = med.priceCurrency.upper() if re.fullmatch(r'[A-Za-z]{3}', med.priceCurrency) else ''
    out['priceCurrency'] = cur if cur and gcount['priceCurrency'][ckey(cur)] >= 3 else ''; P['priceCurrency'] = f'medoid:{med.id}'
    # brand: vote over raw brand strings that map exactly to a canonical brand; fallback title-derived brand key
    bk = Counter(x for x in g.brand_k if x)
    canon = max(bk, key=lambda k: (bk[k], k)) if bk else ''
    raw = [v for v in g.brand if brand_from_field(v) == (canon, 'exact')]
    if raw:
        rc = Counter(raw); out['brand'] = max(rc, key=lambda r: (rc[r], BRAND_RAW_PREF[canon][r], r)); P['brand'] = f'vote:{len(raw)}'
    elif canon:
        out['brand'] = max(BRAND_RAW_PREF[canon], key=lambda r: (BRAND_RAW_PREF[canon][r], r)) if canon in BRAND_RAW_PREF else canon
        P['brand'] = 'title_brand_canonical_spelling'
    else: out['brand'] = ''; P['brand'] = 'none'
    # product type
    pc = Counter(x for x in g.ptype if x in CANON_TYPES)
    out['product_type'] = max(pc, key=lambda k: (pc[k], k)) if pc else ''; P['product_type'] = 'vote'
    pt = out['product_type']
    # model / model_number / chipset / categorical attributes
    for c in ['model', 'model_number', 'chipset_name', 'bus_type', 'interface_type', 'storage_connection_type',
              'memory_type', 'color', 'form_factor']:
        out[c], P[c] = vote_cat(c, g, ctoks, prov)
    if out['model_number'] and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._\-/()]*', out['model_number']):
        out['model_number'] = re.sub(r'[^A-Za-z0-9 ._\-/()]', '', out['model_number']).strip(); P['model_number'] += '+pattern_strip'
    # vram (GPU only): field values (OCR repair) + title-derived value
    if pt == 'GPU':
        cands = [(num(v, True) / 1024 if (num(v, True) or 0) >= 512 else num(v, True), 1) for v in g.vram_gb if v]
        cands += [(float(v), 1) for v in g.vram if v]
        v, s = vote_num([(x, w) for x, w in cands if x and 1 <= x <= 128]); out['vram_gb'] = v if v else ''; P['vram_gb'] = f'vote:{s}'
    else: out['vram_gb'] = ''; P['vram_gb'] = 'n/a'
    # storage (storage types only): field value (TB->GB when the title states the TB figure) + title capacity
    if pt in ('SSD', 'HDD', 'USB_STICK'):
        cands = []
        for r in g.itertuples():
            tc = [x[1] for x in capacities(r.title) if x[1] >= 1]
            fv = num(r.storage_gb)
            if fv:
                if tc and abs(fv * 1000 - tc[0]) / tc[0] < 0.03: fv *= 1000
                elif pt == 'HDD' and fv <= 24: fv *= 1000
                cands.append((fv, 1))
            if tc: cands.append((tc[0], 1))
        v, s = vote_num([(x, w) for x, w in cands if 1 <= x <= 100000], tol=0.03); out['storage_gb'] = v if v else ''; P['storage_gb'] = f'vote:{s}'
        for c in ['read_speed_mb_s', 'write_speed_mb_s']:
            v, s = vote_num([(num(x), 1) for x in g[c] if x and num(x) and 1 <= num(x) <= 20000]); out[c] = v if v else ''; P[c] = f'vote:{s}'
    else:
        for c in ['storage_gb', 'read_speed_mb_s', 'write_speed_mb_s']: out[c] = ''; P[c] = 'n/a'
    # physical dimensions / weight: numeric vote; inches->mm for 3.5"/2.5" drives recorded in inches, kg->g
    for c in ['width_mm', 'length_mm', 'height_mm']:
        cands = []
        for x in g[c]:
            v = num(x)
            if v is None: continue
            if pt == 'HDD' and v < 10: v *= 25.4
            cands.append((v, 1))
        v, s = vote_num(cands); out[c] = round(v, 2) if v else ''; P[c] = f'vote:{s}'
    cands = []
    for x in g['weight_g']:
        v = num(x)
        if v is None: continue
        if (pt in ('HDD', 'GPU') and v < 5) or (pt == 'SSD' and v < 0.1): v *= 1000
        cands.append((v, 1))
    v, s = vote_num([(x, w) for x, w in cands if 0.5 <= x <= 10000]); out['weight_g'] = round(v, 2) if v else ''; P['weight_g'] = f'vote:{s}'
    # schema applicability: GPU-only / storage-only fields
    if pt != 'GPU': out['chipset_name'] = ''
    if pt not in ('SSD', 'HDD', 'USB_STICK'): out['storage_connection_type'] = ''
    if pt not in ('GPU', 'SSD'): out['memory_type'] = ''
    rows.append(out); prov.append(dict(_id=cid, **P))
f = pd.DataFrame(rows)[['_id'] + COLS]
f.to_csv('submission/fused.csv', index=False)
pd.DataFrame(prov).to_csv('work/state/s6_provenance.csv', index=False)
print(f.shape); print((f.replace('', np.nan).notna().mean()).round(3).to_dict())
