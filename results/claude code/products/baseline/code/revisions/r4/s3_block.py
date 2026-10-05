"""Stage 3: blocking + pairwise feature computation.
Hard block: product_type (per-type counts are identical between ds1/ds2, the field is consistent).
Within type, every cross-source pair is scored cheaply (vectorised TF-IDF); the exported candidate set is the
union of (a) top-K neighbours per record per other source by text similarity, (b) shared code tokens."""
import re, os, math, itertools
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

K = 10
MAX_SPEC_BLOCK = 40
d = pd.read_pickle('state/s2_normalized.pkl').reset_index(drop=True)

GENERIC = set('''the and with for in of - graphics card video gpu nvidia amd geforce radeon pci pcie express gen gddr gddr5 gddr6 gddr6x
ddr5 ddr6 ddr3 hdmi dp dvi displayport bit memory ssd hdd hard drive disk solid state internal external usb flash stick pen pendrive
sata iii 3 6gb s gb tb mb rpm cache inch in buy online price new retail oem bulk desktop laptop pc notebook drives disco duro
interne interno festplatte disque dur unidad stick memorie memoria 2.5 3.5 m.2 nvme'''.split())

def tokens(t):
    t = t.lower().replace('™', ' ').replace('®', ' ')
    t = re.sub(r'([a-z])(\d)', r'\1 \2', t)
    t = re.sub(r'(\d)([a-z])', r'\1 \2', t)
    return re.findall(r'[a-z0-9]+(?:\.[0-9]+)?', t)

def codes(r):
    out = set()
    txt = ' '.join([r['n_title'], str(r['model'] or ''), str(r['model_number'] or '')])
    for tok in re.findall(r'[A-Za-z0-9][A-Za-z0-9\-/\.]{3,}[A-Za-z0-9]', txt):
        c = re.sub(r'[^A-Za-z0-9]', '', tok).upper()
        if len(c) >= 6 and re.search(r'[A-Z]', c) and re.search(r'\d', c) and not re.fullmatch(r'\d+(GB|TB|MB|MHZ|RPM|MM|G|T)', c):
            out.add(c)
    if r['n_mpn'] and len(r['n_mpn']) >= 5:
        out.add(r['n_mpn'])
    return out

d['text'] = (d['n_title'] + ' ' + d['model'].fillna('') + ' ' + d['model_number'].fillna(''))
d['tok'] = d['text'].map(tokens)
d['codes'] = d.apply(codes, axis=1)

# variant / series keywords (GPU board partner lines, storage product lines) used as identity evidence
VARIANT = set('''oc dual evo v2 gaming x xs z trio twin fan windforce aorus xtreme master eagle ventus mech armor strix tuf rog phoenix jetstream
turbo blower mini itx amp extreme holo sc ftw3 xc ultra black ko hybrid advanced pulse nitro thicc iii ii red dragon devil
challenger fighter low profile lp silent passive single slot founders edition super ti xt pro plus evo qvo'''.split())

vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, min_df=1)
vec_w = TfidfVectorizer(analyzer=lambda x: x, sublinear_tf=True, min_df=1)
Xc = vec_c.fit_transform(d['text'].str.lower())
Xw = vec_w.fit_transform(d['tok'].map(lambda ts: [t for t in ts if t not in GENERIC]))

rows = []
for ptype, g in d.groupby('product_type'):
    srcs = sorted(g['source'].unique())
    for a, b in itertools.combinations(srcs, 2):
        ia = g.index[g.source == a].values; ib = g.index[g.source == b].values
        Sc = (Xc[ia] @ Xc[ib].T).toarray(); Sw = (Xw[ia] @ Xw[ib].T).toarray()
        S = 0.5 * Sc + 0.5 * Sw
        keep = np.zeros_like(S, dtype=bool)
        k = min(K, S.shape[1]); kk = min(K, S.shape[0])
        top_r = np.argpartition(-S, k - 1, axis=1)[:, :k]
        keep[np.arange(S.shape[0])[:, None], top_r] = True
        top_c = np.argpartition(-S, kk - 1, axis=0)[:kk, :]
        keep[top_c, np.arange(S.shape[1])[None, :]] = True
        ca = d.loc[ia, 'codes'].values; cb = d.loc[ib, 'codes'].values
        code_idx = {}
        for j, cs in enumerate(cb):
            for c in cs: code_idx.setdefault(c, []).append(j)
        for i, cs in enumerate(ca):
            for c in cs:
                for j in code_idx.get(c, []): keep[i, j] = True
        # spec-key blocks: same brand family + GPU chip / storage capacity (complements text top-K for crowded families)
        def spec_key(r):
            if not r['n_brand_fam']: return None
            if r['product_type'] == 'GPU': return (r['n_brand_fam'], r['n_chip']) if r['n_chip'] else None
            return (r['n_brand_fam'], r['n_storage_gb']) if r['n_storage_gb'] == r['n_storage_gb'] else None
        ka = [spec_key(r) for r in d.loc[ia].to_dict('records')]; kb = [spec_key(r) for r in d.loc[ib].to_dict('records')]
        kidx = {}
        for j, k in enumerate(kb):
            if k is not None: kidx.setdefault(k, []).append(j)
        for i, k in enumerate(ka):
            if k is not None and len(kidx.get(k, [])) <= MAX_SPEC_BLOCK:
                for j in kidx.get(k, []): keep[i, j] = True
        ii, jj = np.nonzero(keep)
        for i, j in zip(ii, jj):
            rows.append((ia[i], ib[j], ptype, Sc[i, j], Sw[i, j]))
cand = pd.DataFrame(rows, columns=['i', 'j', 'ptype', 'sim_char', 'sim_word'])

def mpn_eq(a, b):
    if a == b: return True
    p = os.path.commonprefix([a, b])
    # regional / packaging suffixes (e.g. MZ-76E250B vs MZ-76E250BW, MU-PA1T0B/WW vs /AM): letters only after >=7 shared chars
    return len(p) >= 7 and re.fullmatch(r'[A-Z]{0,4}', a[len(p):]) is not None and re.fullmatch(r'[A-Z]{0,4}', b[len(p):]) is not None

def mpn_rel(a, b, code):
    a2, b2 = a.replace('O', '0').replace('I', '1'), b.replace('O', '0').replace('I', '1')
    if mpn_eq(a2, b2): return 1
    if code: return 0
    if len(a2) >= 4 and len(b2) >= 4 and (a2 in b2 or b2 in a2): return 0   # e.g. DT50 vs DT50/32GB, CZ73 vs SDCZ73-064G
    # only a conflict when both codes come from the same naming scheme (shared prefix), e.g. WD80PURZ vs WD82PURZ;
    # unrelated schemes (ASUS 90YV0... vs DUAL-RTX...) are uninformative
    return -1 if len(os.path.commonprefix([a2, b2])) >= 2 else 0

def feat(r1, r2):
    f = {}
    f['code'] = int(bool(r1['codes'] & r2['codes']))
    c1, c2 = r1['n_storage_gb'], r2['n_storage_gb']
    f['cap'] = 0 if (c1 != c1 or c2 != c2) else (1 if abs(c1 - c2) <= 0.02 * max(c1, c2) else -1)
    v1, v2 = r1['n_vram_gb'], r2['n_vram_gb']
    f['vram'] = 0 if (v1 != v1 or v2 != v2) else (1 if v1 == v2 else -1)
    h1, h2 = r1['n_chip'], r2['n_chip']
    f['chip'] = 0 if (not h1 or not h2) else (1 if h1 == h2 else -1)
    b1, b2 = r1['n_brand_fam'], r2['n_brand_fam']
    f['brand'] = 0 if (not b1 or not b2) else (1 if b1.lower() == b2.lower() else -1)
    m1, m2 = r1['n_mem'], r2['n_mem']
    m1, m2 = MEMEQ.get(m1, m1), MEMEQ.get(m2, m2)
    f['mem'] = 0 if (m1 not in MEMC or m2 not in MEMC or r1['product_type'] != 'GPU') else (1 if m1 == m2 else -1)
    f['excl'] = excl_conflict(r1['tokset'], r2['tokset'])
    t1 = set(r1['tok']) & VARIANT; t2 = set(r2['tok']) & VARIANT
    f['var_common'] = len(t1 & t2); f['var_diff'] = len(t1 ^ t2)
    m1, m2 = r1['n_mpn'], r2['n_mpn']
    f['mpn'] = 0 if (not m1 or not m2) else mpn_rel(m1, m2, f['code'])
    if r1['product_type'] != 'GPU':
        a, b = r1['mtok'], r2['mtok']
        core = lambda x: re.sub(r'[a-z]', '', x)
        f['mtok'] = 0 if (not a or not b) else (1 if a & b else (0 if any(core(x) == core(y) for x in a for y in b) else -1))
        f['sasg'] = 0 if (not r1['sasg'] or not r2['sasg']) else (1 if r1['sasg'] == r2['sasg'] else -1)
        rp = {r1['rpm'].replace('5900', '5400'), r2['rpm'].replace('5900', '5400')}   # 5400/5900 "class" drives are listed both ways
        f['rpm'] = 0 if (not r1['rpm'] or not r2['rpm']) else (1 if len(rp) == 1 else -1)
        a, b = r1['ikind'], r2['ikind']
        f['iface'] = 0 if (not a or not b) else (1 if a & b else -1)
        a, b = r1['fkind'], r2['fkind']
        f['ffk'] = 0 if (not a or not b) else (1 if a & b else -1)
        a, b = r1['lines'], r2['lines']
        f['line_common'] = len(a & b); f['line_diff'] = len(a ^ b) if (a and b) else 0
        # asymmetric tier modifiers on a shared product line (IronWolf vs IronWolf Pro, Red vs Red Plus)
        base = (a & b) - MODS
        f['mod_diff'] = int(bool(base) and bool((r1['tokset'] ^ r2['tokset']) & MODS))
    else:
        f['mod_diff'] = 0
        f['mtok'] = f['sasg'] = f['rpm'] = f['iface'] = f['ffk'] = f['line_common'] = f['line_diff'] = 0
    return f

UNIT_TOK = re.compile(r'^\d+(\.\d+)?(gb|tb|mb|g|t|go|to|rpm|mhz|mm|bit|k|x|w|gbps|mbps|gbs|in|inch|pcs|p|hz|yrs?|yr)$')
def model_tokens(r):
    # alphanumeric model tokens (e.g. 860, 970, sn550, mx500, p400, dc500m) from title+model, before letter/digit split
    txt = (r['n_title'] + ' ' + str(r['model'] or '')).lower()
    out = set()
    txt = re.sub(r'\b\d+(\.\d+)?\s*(mb/s|mo/s|mbps|mb/sec|gb/s)', ' ', txt)
    for tk in re.findall(r'[a-z0-9]+', txt):
        if UNIT_TOK.match(tk) or tk in ('2280', '2260', '2242', '2230', '3d', '4k', '8k', '2020', '2019', '2021', '2018', '1080p'): continue
        if re.fullmatch(r'[a-z]{0,4}\d{2,4}[a-z]{0,3}', tk) and re.search(r'\d{3}', tk):
            if re.fullmatch(r'\d+', tk) and (float(tk) in (r['n_storage_gb'], r['n_storage_gb'] / 1000 if r['n_storage_gb'] == r['n_storage_gb'] else -1) or int(tk) in (5400, 5900, 7200, 10000, 15000, 7200)): continue
            out.add(tk)
    return out
def rpm(r):
    txt = (r['n_title'] + ' ' + r['n_desc'][:300]).lower()
    m = re.search(r'\b(5400|5900|7200|10000|15000)\s*-?rpm|\b(10k|15k|7\.2k|5\.4k|10\.5k|7200|5400|5900)\b', txt)
    if not m: return ''
    v = m.group(1) or m.group(2)
    return {'10k': '10000', '15k': '15000', '7.2k': '7200', '5.4k': '5400', '10.5k': '10500'}.get(v, v)
def iface_kind(r):
    txt = (r['n_title'] + ' ' + r['n_iface'] + ' ' + str(r['n_bus'])).lower()
    ks = set()
    if 'nvme' in txt or 'pcie' in txt: ks.add('nvme')
    if re.search(r'\bsas\b', txt): ks.add('sas')
    if 'sata' in txt: ks.add('sata')
    if 'usb' in txt: ks.add('usb')
    return ks
MEMC = {'GDDR3','GDDR4','GDDR5X','GDDR5','GDDR6X','GDDR6','GDDR7','HBM2','HBM','DDR3','DDR4','DDR5','LPDDR4'}
MEMEQ = {'DDR3': 'GDDR3', 'DDR5': 'GDDR5'}   # GPU listings use DDR3/GDDR3 interchangeably
def sas_gen(r):
    txt = r['n_title'].lower()
    if 'sas' not in txt: return ''
    m = re.search(r'\b(3|6|12)\s*g(b|bps|b/s|bit/s)?\b', txt)
    return m.group(1) if m else ''
def ff_kind(r):
    txt = r['n_title'].lower().replace(',', '.')   # title only: the extracted form-factor field is noisy
    ks = set()
    if re.search(r'2\.5|\bsff\b', txt): ks.add('2.5')
    if re.search(r'3\.5|\blff\b', txt): ks.add('3.5')
    if re.search(r'\bm\.?2\b', txt): ks.add('m2')
    return ks
LINES = set('''blue red black green purple gold white barracuda ironwolf exos skyhawk firecuda constellation cheetah savvio nytro
evo pro qvo plus portable passport elements expansion backup book essentials extreme ultra cruzer blade glide flair fit
datatraveler kyson a400 mx500 bx500 p1 p2 p5 x6 x8 n300 x300 p300 l200 mg04 mg06 mg07 mg08 hc510 ultrastar deskstar
vault privacy locker se9 exodia savage fury hyperx canvas'''.split())
d['mtok'] = d.apply(model_tokens, axis=1)
d['rpm'] = d.apply(rpm, axis=1)
d['ikind'] = d.apply(iface_kind, axis=1)
d['fkind'] = d.apply(ff_kind, axis=1)
d['sasg'] = d.apply(sas_gen, axis=1)
d['lines'] = d['tok'].map(lambda ts: set(ts) & LINES)

# mutually exclusive product-line / board-partner series vocabularies: a pair naming different members of the
# same group is a contradiction (e.g. WD Blue vs WD Purple, ASUS Strix vs ASUS Dual, Seagate Exos vs Archive)
EXCL_GROUPS = [
    set('blue red black green purple gold'.split()),
    set('barracuda ironwolf exos skyhawk firecuda archive constellation cheetah savvio nytro'.split()),
    set('evo qvo pro'.split()),
    set('strix dual tuf turbo phoenix cerberus windforce aorus eagle vision ventus mech armor trio aero suprim amp twin trinity holo sc xc ftw3 kingpin pulse nitro thicc stormx jetstream gamingpro xlr8 mini'.split()),
    set('passport elements expansion book essentials'.split()),
    set('cruzer ultra extreme glide blade flair fit'.split()),
]
MODS = {'pro', 'plus', 'max', 'go'}
def excl_conflict(t1, t2):
    for g in EXCL_GROUPS:
        a, b = t1 & g, t2 & g
        if a and b and not (a & b): return 1
    return 0
d['tokset'] = d['tok'].map(set)
recs = d.to_dict('records')
F = pd.DataFrame([feat(recs[i], recs[j]) for i, j in zip(cand.i, cand.j)])
cand = pd.concat([cand.reset_index(drop=True), F], axis=1)
cand['id1'] = d.loc[cand.i, 'id'].values; cand['id2'] = d.loc[cand.j, 'id'].values
cand['src1'] = d.loc[cand.i, 'source'].values; cand['src2'] = d.loc[cand.j, 'source'].values
cand.to_pickle('state/s3_candidates.pkl')
d.to_pickle('state/s3_records.pkl')
print(len(cand), cand.groupby(['src1', 'src2']).size().to_dict())
