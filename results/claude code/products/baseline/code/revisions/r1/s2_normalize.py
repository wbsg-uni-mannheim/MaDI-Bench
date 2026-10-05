"""Stage 2: normalization. Adds canonical columns (n_*) next to raw values."""
import re, math
import pandas as pd, numpy as np

d = pd.read_pickle('state/s1_translated.pkl')

def s(x):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else str(x).strip()

# ---------- brand ----------
BRAND_ALIAS = {
    'wd': 'Western Digital', 'western digital': 'Western Digital', 'wester digital': 'Western Digital',
    'westerndigital': 'Western Digital', 'western digital (wd)': 'Western Digital',
    'hewlett packard enterprise': 'HPE', 'hpe': 'HPE', 'hp enterprise': 'HPE',
    'hp': 'HP', 'hewlett packard': 'HP', 'hewlett-packard': 'HP',
    'kingston technology': 'Kingston', 'kingston': 'Kingston',
    'a-data': 'ADATA', 'adata': 'ADATA', 'adata technology': 'ADATA',
    'sapphire technology': 'Sapphire', 'sapphire': 'Sapphire',
    'sandisk': 'SanDisk', 'samsung': 'Samsung', 'seagate': 'Seagate', 'gigabyte': 'Gigabyte',
    'asus': 'ASUS', 'zotac': 'ZOTAC', 'msi': 'MSI', 'toshiba': 'Toshiba', 'intel': 'Intel',
    'crucial': 'Crucial', 'corsair': 'Corsair', 'evga': 'EVGA', 'pny': 'PNY', 'palit': 'Palit',
    'patriot': 'Patriot', 'lacie': 'LaCie', 'xpg': 'XPG', 'silicon power': 'Silicon Power',
    'micron': 'Micron', 'xfx': 'XFX', 'amd': 'AMD', 'nvidia': 'NVIDIA', 'team group': 'Team Group',
    'teamgroup': 'Team Group', 'team': 'Team Group', 'dell': 'Dell', 'lenovo': 'Lenovo',
    'powercolor': 'PowerColor', 'inno3d': 'Inno3D', 'transcend': 'Transcend', 'lexar': 'Lexar',
    'exos': 'Seagate', 'barracuda': 'Seagate', 'ironwolf': 'Seagate', 'firecuda': 'Seagate', 'hikvision seagate': 'Seagate',
    'maxtor/seagate': 'Seagate', 'sg': 'Seagate', 'seagate lacie': 'LaCie', 'la cie': 'LaCie', 'radeon': 'AMD',
    'wd_black': 'Western Digital', 'wd western digital': 'Western Digital', 'hyperx': 'Kingston', 'kingston digital': 'Kingston',
    'hewlett packard hp': 'HP', 'patriot memory': 'Patriot', 'adata xpg': 'XPG', 'zotac gaming': 'ZOTAC', 'aorus': 'Gigabyte',
    'rog strix': 'ASUS', '华硕': 'ASUS', 'asrock': 'ASRock', 'g-technology': 'G-Technology',
    'verbatim': 'Verbatim', 'ibm': 'IBM', 'hgst': 'HGST', 'fujitsu': 'Fujitsu',
}
def norm_brand(b):
    b0 = s(b)
    if not b0: return ''
    k = re.sub(r'[®™]', '', b0).lower().strip()
    k = re.sub(r'\s+(inc\.?|corp\.?|corporation|ltd\.?|co\.?)$', '', k).strip()
    return BRAND_ALIAS.get(k, b0)
# brand family key for matching compatibility (HP/HPE and ADATA/XPG sub-brand are treated as compatible)
# NVIDIA/AMD as a brand on a graphics card is the chip vendor (reference/workstation cards sold by partners), so it is
# treated as unknown for matching compatibility.
BRAND_FAMILY = {'HPE': 'HP', 'HP': 'HP', 'XPG': 'ADATA', 'ADATA': 'ADATA', 'NVIDIA': '', 'AMD': ''}

# ---------- text ----------
def clean_text(t):
    t = s(t)
    t = t.replace('\\"', '"').replace('\\', '')
    t = re.sub(r'[®™©]', ' ', t)
    return t

# ---------- capacity ----------
CAP_RE = re.compile(r'(?<![\w.])(\d+(?:[.,]\d{1,2})?)\s*-?\s*(tb|gb|to|go|t|g)(?![a-z0-9/])', re.I)
def caps_in(text):
    out = []
    for m in CAP_RE.finditer(text):
        num = float(m.group(1).replace(',', '.'))
        u = m.group(2).lower()
        after = text[m.end():m.end()+6].lower()
        before = text[max(0, m.start()-6):m.start()].lower()
        if u == 'g' and num < 16: continue  # bare 'G' below 16 is usually an interface speed (6G/12G SAS)
        if u == 'g' and re.match(r'\s*(b/s|bps|bit)', after): continue
        gb = num * 1000 if u in ('tb', 'to', 't') else num
        if gb <= 0 or gb > 30000: continue
        out.append(gb)
    return out

def round_cap(gb):
    if gb is None or (isinstance(gb, float) and math.isnan(gb)): return np.nan
    if gb in (1024, 2048, 4096, 8192): return gb / 1024 * 1000
    return round(gb, 2)

def resolve_capacity(r):
    f = r['storage_gb']
    f = None if f is None or (isinstance(f, float) and math.isnan(f)) else float(f)
    if r['product_type'] == 'GPU':
        return np.nan, 'na'
    Tset = sorted(set(round_cap(c) for c in caps_in(r['n_title'])))
    if f is not None:
        if round_cap(f) in Tset: return round_cap(f), 'field=title'
        if round_cap(f*1000) in Tset: return round_cap(f*1000), 'field*1000=title'
        if len(Tset) == 1: return Tset[0], 'title_override'
        D = sorted(set(round_cap(c) for c in caps_in(r['n_desc'])))
        if round_cap(f) in D: return round_cap(f), 'field=desc'
        if round_cap(f*1000) in D: return round_cap(f*1000), 'field*1000=desc'
        # bare small values for HDD/SSD are TB (e.g. 6.0 for a 6TB drive)
        if f < 20 and r['product_type'] in ('HDD', 'SSD'): return round_cap(f*1000), 'field*1000_heur'
        return round_cap(f), 'field'
    if len(Tset) == 1: return Tset[0], 'title_only'
    return np.nan, 'missing' if not Tset else 'ambiguous_title'

# ---------- GPU ----------
def norm_chip(x):
    t = clean_text(x).lower()
    t = re.sub(r'\b(nvidia|amd|geforce|radeon|graphics|card)\b', ' ', t)
    return re.sub(r'[^a-z0-9]+', '', t)
def chip_from_title(t):
    t2 = re.sub(r'\b(nvidia|amd|geforce|radeon)\b', ' ', t.lower())
    m = re.search(r'\b(rtx|gtx|gt|rx)\s*-?(\d{3,4})\s*(ti|super|xt)?\b(\s*(super))?', t2)
    if m:
        return (m.group(1) + m.group(2) + (m.group(3) or '') + (m.group(5) or '')).lower()
    m = re.search(r'\bquadro\s*(rtx\s*)?([a-z]{0,2}\d{2,4})\b', t2)
    if m: return 'quadro' + (m.group(1) or '').replace(' ', '') + m.group(2)
    m = re.search(r'\b(pro\s*)?wx\s*(\d{4})\b', t2)
    if m: return 'prowx' + m.group(2)
    return ''
def gpu_chip(r):
    if r['product_type'] != 'GPU': return ''
    c = norm_chip(r['chipset_name'])
    t = chip_from_title(r['n_title'])
    return c or t
def gpu_vram(r):
    if r['product_type'] != 'GPU': return np.nan
    v = r['vram_gb']
    v = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    if v is not None and v >= 256: v = v / 1024
    if v is not None: return float(round(v)) if v >= 1 else v
    T = sorted(set(x for x in caps_in(r['n_title']) if x <= 48))
    return T[0] if len(T) == 1 else np.nan

# ---------- memory type ----------
MEM_OK = ['GDDR3','GDDR4','GDDR5X','GDDR5','GDDR6X','GDDR6','GDDR7','HBM2e','HBM2','HBM3e','HBM3','HBM',
          'LPDDR4X','LPDDR4','LPDDR5X','LPDDR5','DDR3L','DDR3','DDR4','DDR5']
def norm_mem(x, ptype):
    t = s(x)
    if not t: return ''
    u = re.sub(r'[\s\-]', '', t.upper())
    u = u.replace('GDRR', 'GDDR').replace('HIGHBANDWIDTHMEMORY2', 'HBM2')
    if ptype == 'GPU' and re.match(r'^DDR[56]', u): u = 'G' + u
    for m in MEM_OK:
        if m.upper() in u: return m
    if 'TLC' in u: return 'TLC'
    if 'MLC' in u: return 'MLC'
    if 'NAND' in u: return 'NAND'
    return t

# ---------- bus ----------
def norm_bus(x):
    t = s(x)
    if not t: return ''
    u = t.upper().replace('-', ' ').replace('_', ' ')
    u = re.sub(r'\s+', ' ', u)
    if 'THUNDERBOLT' in u:
        m = re.search(r'(\d)', u); return 'Thunderbolt' + (' ' + m.group(1) if m else '')
    if 'USB' in u:
        if re.search(r'3\.2\s*GEN\s*2\s*X\s*2', u): return 'USB 3.2 Gen 2x2'
        if re.search(r'3\.[12]\s*GEN\s*2', u): return 'USB 3.2 Gen 2'
        if re.search(r'3\.[12]\s*GEN\s*1', u) or re.search(r'USB ?3\.0', u): return 'USB 3.0'
        if re.search(r'3\.2', u): return 'USB 3.2'
        if re.search(r'3\.1', u): return 'USB 3.1'
        if re.search(r'USB ?3\b', u): return 'USB 3.0'
        if re.search(r'2\.0', u): return 'USB 2.0'
        if re.search(r'TYPE ?C|USB ?C', u): return 'USB-C'
        return 'USB'
    if 'SAS' in u or 'SERIAL ATTACHED' in u:
        m = re.search(r'(12|6|3|24)\s*G', u)
        return 'SAS ' + m.group(1) + 'Gb/s' if m else 'SAS'
    if 'SATA' in u or 'SERIAL ATA' in u:
        if re.search(r'6\s*\.?0?\s*G|III|SATA ?3|600', u): return 'SATA 6Gb/s'
        if re.search(r'3\s*\.?0?\s*G|\bII\b|SATA ?2', u): return 'SATA 3Gb/s'
        return 'SATA'
    if 'PCI' in u or 'NVME' in u:
        u2 = re.sub(r'X ?(16|8|4|1)\b', ' ', u)
        g = re.search(r'(?:GEN ?|PCIE ?|EXPRESS ?|PCI E ?|\b)([2345])(?:\.0)?\b', u2)
        l = re.search(r'X ?(16|8|4|1)\b', u)
        out = 'PCIe'
        if g: out += ' %s.0' % g.group(1)
        if l: out += ' x%s' % l.group(1)
        return out
    return t

# ---------- interface ----------
def norm_iface(x):
    t = s(x)
    if not t: return ''
    u = t.upper()
    if 'NVME' in u: return 'NVMe'
    if 'MSATA' in u: return 'mSATA'
    if 'SAS' in u or 'SERIAL ATTACHED' in u: return 'SAS'
    if 'SATA' in u or 'SERIAL ATA' in u: return 'SATA'
    if 'LIGHTNING' in u: return 'Lightning'
    if 'MICRO' in u and 'USB' in u: return 'Micro USB'
    if re.search(r'TYPE[\s-]*C|USB[\s-]*C\b', u) and not re.search(r'TYPE[\s-]*A', u): return 'USB-C'
    if 'USB' in u or 'TYPE' in u: return 'USB'
    if 'PCI' in u: return 'PCIe'
    return t

# ---------- form factor ----------
def norm_ff(x):
    t = s(x)
    if not t: return ''
    u = t.lower().replace(',', '.')
    if 'msata' in u: return 'mSATA'
    if 'm.2' in u or re.search(r'\bm2\b', u) or re.fullmatch(r'22\d\d', u.strip()): return 'M.2'
    if re.search(r'2\.5', u) or 'sff' in u: return '2.5-inch'
    if re.search(r'3\.5', u) or 'lff' in u: return '3.5-inch'
    if 'low profile' in u or 'low-profile' in u: return 'Low Profile'
    if 'atx' in u: return 'ATX'
    if 'external' in u or 'portable' in u: return 'External'
    return t

def norm_conn(x):
    t = s(x)
    if not t: return ''
    ff = norm_ff(t)
    ff = ff if ff in ('2.5-inch', '3.5-inch', 'M.2', 'mSATA') else ''
    ifc = norm_iface(t)
    ifc = ifc if ifc in ('SATA', 'SAS', 'NVMe', 'USB', 'USB-C', 'Lightning', 'Micro USB', 'PCIe') else ''
    if ff == 'mSATA': ifc = ''
    out = ' '.join(p for p in (ff, ifc) if p)
    return out or t

COLOR = {'negru': 'Black', 'zwart': 'Black', 'sort': 'Black', 'dyb sort': 'Black', 'noir': 'Black', 'schwarz': 'Black',
         'nero': 'Black', 'czarny': 'Black', 'zilver': 'Silver', 'argintiu': 'Silver', 'silber': 'Silver', 'argent': 'Silver',
         'rosu': 'Red', 'rood': 'Red', 'rot': 'Red', 'rouge': 'Red', 'albastru': 'Blue', 'blauw': 'Blue', 'blau': 'Blue', 'bleu': 'Blue',
         'alb': 'White', 'wit': 'White', 'weiss': 'White', 'weiß': 'White', 'blanc': 'White', 'hvid': 'White',
         'grijs': 'Grey', 'gri': 'Grey', 'grå': 'Grey', 'silver colour': 'Silver', 'groen': 'Green', 'verde': 'Green'}
def norm_color(x):
    t = s(x)
    if not t: return ''
    k = t.lower().strip()
    if k in COLOR: return COLOR[k]
    if t.isupper() or t.islower():
        return ' '.join(w.capitalize() for w in k.split())
    return t

CUR = {'лв.': 'BGN', 'LEI': 'RON', 'Kč': 'CZK', '€': 'EUR', '$': 'USD', '£': 'GBP', 'zł': 'PLN'}
def norm_cur(x):
    t = s(x)
    t = CUR.get(t, t)
    return t.upper() if re.fullmatch(r'[A-Za-z]{3}', t) else ''
def norm_price(x):
    t = s(x).replace(',', '')
    try:
        v = float(t)
    except ValueError:
        return np.nan
    return v if 0 < v < 99999 else np.nan

def num(x):
    try:
        v = float(x)
        return np.nan if math.isnan(v) else v
    except (TypeError, ValueError):
        return np.nan

def norm_mpn(x):
    return re.sub(r'[^A-Za-z0-9]', '', s(x)).upper()

d['n_title'] = d['title'].map(clean_text)
def infer_type(r):
    if s(r['product_type']) in ('GPU', 'SSD', 'HDD', 'USB_STICK'): return r['product_type']
    tl = r['n_title'].lower()
    if re.search(r'flash drive|usb stick|pen ?drive|otg', tl): return 'USB_STICK'
    if re.search(r'\bssd\b|solid state', tl): return 'SSD'
    if re.search(r'graphics|geforce|radeon|gpu', tl): return 'GPU'
    if re.search(r'hdd|hard (disk|drive)', tl): return 'HDD'
    return ''
d['product_type_raw'] = d['product_type']
d['product_type'] = d.apply(infer_type, axis=1)
d['n_desc'] = d['description'].map(clean_text)
d['n_brand'] = d['brand'].map(norm_brand)
known = sorted(set(BRAND_ALIAS.values()) | (set(d['n_brand'].unique()) - {''}), key=len, reverse=True)
def brand_fb(r):
    if r['n_brand']: return r['n_brand'], 'field'
    tl = r['n_title'].lower()
    for b in known:
        if re.search(r'\b' + re.escape(b.lower()) + r'\b', tl): return b, 'title'
    for a, b in BRAND_ALIAS.items():
        if len(a) > 2 and re.search(r'\b' + re.escape(a) + r'\b', tl): return b, 'title'
    return '', 'missing'
bb = d.apply(brand_fb, axis=1)
d['n_brand'] = [x[0] for x in bb]; d['n_brand_src'] = [x[1] for x in bb]
d['n_brand_fam'] = d['n_brand'].map(lambda b: BRAND_FAMILY.get(b, b))
cc = d.apply(resolve_capacity, axis=1)
d['n_storage_gb'] = [x[0] for x in cc]; d['n_storage_src'] = [x[1] for x in cc]
d['n_chip'] = d.apply(gpu_chip, axis=1)
d['n_vram_gb'] = d.apply(gpu_vram, axis=1)
d['n_mem'] = [norm_mem(x, p) for x, p in zip(d['memory_type'], d['product_type'])]
d['n_bus'] = d['bus_type'].map(norm_bus)
d['n_iface'] = d['interface_type'].map(norm_iface)
d['n_ff'] = d['form_factor'].map(norm_ff)
d['n_conn'] = d['storage_connection_type'].map(norm_conn)
d['n_color'] = d['color'].map(norm_color)
d['n_cur'] = d['priceCurrency'].map(norm_cur)
d['n_price'] = d['price'].map(norm_price)
d['n_mpn'] = d['model_number'].map(norm_mpn)
for c in ['read_speed_mb_s', 'write_speed_mb_s', 'width_mm', 'length_mm', 'height_mm', 'weight_g']:
    d['n_' + c] = d[c].map(num)
# speeds given in GB/s (e.g. 3.5) -> MB/s
for c in ['n_read_speed_mb_s', 'n_write_speed_mb_s']:
    d[c] = d[c].map(lambda v: v * 1000 if v == v and v < 10 else v)
d.to_pickle('state/s2_normalized.pkl')

rows = []
NC = {'brand': 'n_brand', 'storage_gb': 'n_storage_gb', 'vram_gb': 'n_vram_gb', 'chipset_name': 'n_chip', 'memory_type': 'n_mem',
      'bus_type': 'n_bus', 'interface_type': 'n_iface', 'form_factor': 'n_ff', 'storage_connection_type': 'n_conn', 'color': 'n_color'}
CANON_SETS = {'n_ff': {'2.5-inch', '3.5-inch', 'M.2', 'mSATA', 'Low Profile', 'ATX', 'External'},
              'n_iface': {'SATA', 'NVMe', 'SAS', 'USB', 'USB-C', 'Lightning', 'Micro USB', 'mSATA', 'PCIe'},
              'n_mem': set(MEM_OK) | {'TLC', 'MLC', 'NAND'}}
for c, nc in NC.items():
    for src, g in d.groupby('source'):
        raw = g[c].map(s).ne('').sum(); canon = g[nc].map(lambda v: s(v) != '').sum()
        inset = g[nc].isin(CANON_SETS[nc]).sum() if nc in CANON_SETS else None
        rows.append(dict(source=src, attribute=c, raw_non_null=int(raw), canonical_non_null=int(canon),
                         in_canonical_vocab=inset))
pd.DataFrame(rows).to_csv('taxonomy_coverage.csv', index=False)
print(d['n_storage_src'].value_counts().to_dict())
print((d[d.product_type == 'GPU'].n_chip == '').sum(), 'GPU without chip')
