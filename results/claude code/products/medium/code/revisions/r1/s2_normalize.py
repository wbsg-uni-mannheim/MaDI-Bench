"""Stage 2: normalization. Reads work/state/translated.csv, writes work/state/normalized.pkl/.csv
with raw columns kept and canonical/derived columns prefixed n_ / f_."""
import pandas as pd, re, json, os, difflib
from collections import Counter
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)

NULLS = {'', 'n/a', 'na', 'n/z', 'n/s', 'n/w', 'none', 'null', 'nan', '-', 'o', 'unchanged'}

# ---------------- numbers ----------------
def parse_num(s, thousands_bias=False):
    """Locale-aware number parse with OCR repair. Returns float or None."""
    if s is None: return None
    s = str(s).strip()
    if s.lower().replace(' ', '') in NULLS: return None
    s = re.sub(r'(?<=[\d.,])\s+(?=[\d.,])', '', s)          # '20 00.0' -> '2000.0'
    s = re.sub(r'(?<=[\d.,])[Oo]|[Oo](?=[\d.,])', '0', s)    # OCR O->0
    s = re.sub(r'(?<=[\d.,])[Oo]', '0', s)
    s = re.sub(r'(?<=[\d.,])[Il]', '1', s)
    s = re.sub(r'(?<=[\d.,])[S]', '5', s)
    m = re.search(r'[\d.,]*\d', s)
    if not m: return None
    tail = s[m.end():].strip()
    if tail and re.match(r'^[A-Za-z]$', tail):  # stray OCR char like '2.p', '189.9p'
        return None if s[m.end()-1] in '.,' else None
    num = m.group(0)
    if re.fullmatch(r'\d+', num): return float(num)
    has_c, has_d = ',' in num, '.' in num
    if has_c and has_d:
        if num.rfind(',') > num.rfind('.'):
            num = num.replace('.', '').replace(',', '.')
        else:
            num = num.replace(',', '')
    else:
        sep = ',' if has_c else '.'
        parts = num.split(sep)
        if len(parts) > 2:
            num = num.replace(sep, '')
        else:
            if len(parts[1]) == 3 and thousands_bias:
                num = parts[0] + parts[1]
            else:
                num = parts[0] + '.' + parts[1]
    try: return float(num)
    except ValueError: return None

# ---------------- product type ----------------
PT_ALIASES = {
    'GPU': ['gpu', 'g pu', 'gp u', '6pu', 'hpu', 'graphics processing unit', 'graphical processing unit'],
    'SSD': ['ssd', 'ss d', 's5d', 'asd', 'solid state drive', 'u.2 ssd', 'nvme ssd', 'portable_ssd', 'portabls_ssd'],
    'HDD': ['hdd', 'h dd', 'hdx', 'hxd', 'udd', 'hard disk drive', 'external hdd'],
    'USB_STICK': ['usb_stick', 'usb_st1ck', 'usb_sfick', 'usb flash drive', 'usb stick', 'usb memory stick', 'drive flash usb'],
}
PT_MAP = {a: k for k, v in PT_ALIASES.items() for a in v}
def norm_pt(s):
    k = re.sub(r'\s+', ' ', s.strip().lower())
    if not k: return None
    if k in PT_MAP: return PT_MAP[k]
    # other categories: collapse spacing/case, repair OCR digits, canonical UPPER_SNAKE
    k2 = re.sub(r'[^a-z0-9]+', '_', k).strip('_').upper()
    return k2

OTHER_FIX = {'MON8TOR': 'MONITOR', 'MON9TOR': 'MONITOR', 'MONIT0R': 'MONITOR', 'KONITOR': 'MONITOR',
             'MON_ITOR': 'MONITOR', 'MONI_TOR': 'MONITOR', 'MONITO_R': 'MONITOR', 'M0NITOR': 'MONITOR',
             'MONITOR_DISPLAY': 'MONITOR', 'STORAGE_INTERFZCE': 'STORAGE_INTERFACE', 'INTERFACE_STORAGE': 'STORAGE_INTERFACE',
             'NETWORK_SWIGCH': 'NETWORK_SWITCH', 'MEMORY_MO_DULE': 'MEMORY_MODULE', 'ROUTCR': 'ROUTER',
             'PORTABLE_AUDIO_REEORDER': 'PORTABLE_AUDIO_RECORDER', 'MONITORING_CONTROL_LER': 'MONITORING_CONTROLLER',
             'KEYBOARD_DOCK': 'KEYBOARDDOCK', 'CONTROLLER_MIDI': 'MIDI_CONTROLLER', 'EAM': 'RAM'}

def infer_pt(title):
    s = title.lower()
    if re.search(r'graphics? card|video card|geforce|radeon|quadro|\bgtx\b|\brtx\b|placa video|\bvga\b|grafikkarte', s): return 'GPU'
    if re.search(r'flash drive|usb stick|pen ?drive|cruzer|datatraveler|data traveler|flash memory|ixpand|flashdrive|memorie usb|cl[ée] usb|usb-stick|memory stick|\bdt\d', s): return 'USB_STICK'
    if re.search(r'\bssd\b|solid state|nvme|optane|m\.2', s): return 'SSD'
    if re.search(r'\bhdd\b|hard dis[kc]|hard drive|harddisk|\brpm\b|ironwolf|barracuda|disco|\bsas\b|backup plus|exos', s): return 'HDD'
    return None

# ---------------- brand ----------------
BRAND_ALIAS = {'wd': 'Western Digital', 'w d': 'Western Digital', 'western digital': 'Western Digital', 'wd western digital': 'Western Digital',
    'western digital corporation': 'Western Digital', 'kingston technology': 'Kingston', 'patriot memory': 'Patriot',
    'la cie': 'LaCie', 'lacie sas': 'LaCie', 'a-data': 'ADATA', 'adata xpg': 'ADATA', 'xpg': 'ADATA', 'hpe': 'HP', 'hewlett packard enterprise': 'HP',
    'seagate technology': 'Seagate', 'samsung electronics': 'Samsung', 'gigabyte technology': 'Gigabyte', 'aorus': 'Gigabyte',
    'team': 'Team Group', 'teamgroup': 'Team Group', 'intel®': 'Intel', 'sapphire technology': 'Sapphire', 'rog strix': 'ASUS',
    'hyperx': 'Kingston', 'dell equallogic': 'Dell', 'g-technology': 'G-Technology'}
REAL_BRANDS = ['Seagate', 'Western Digital', 'Gigabyte', 'Samsung', 'ASUS', 'Kingston', 'SanDisk', 'Toshiba', 'ADATA', 'HP', 'Crucial',
    'Sapphire', 'Intel', 'MSI', 'Palit', 'PNY', 'ZOTAC', 'EVGA', 'Silicon Power', 'Patriot', 'Corsair', 'LaCie', 'AMD', 'Transcend',
    'Dell', 'PowerColor', 'NVIDIA', 'Micron', 'XFX', 'Lenovo', 'Team Group', 'iStorage', 'Intenso', 'Maxtor', 'Lexar', 'Fujitsu',
    'Hitachi', 'HGST', 'Verbatim', 'Sony', 'Toshiba', 'Inno3D', 'Gainward', 'KFA2', 'Galax', 'ASRock', 'IBM', 'Netac', 'Lenovo', 'Apple', 'Integral', 'PQI', 'Emtec']
REAL_LC = {b.lower(): b for b in REAL_BRANDS}
TITLE_BRAND_PAT = [(re.compile(r'\b' + re.escape(k) + r'\b', re.I), v) for k, v in
    sorted(list(REAL_LC.items()) + [('wd', 'Western Digital'), ('western digital', 'Western Digital'), ('a-data', 'ADATA'),
     ('hpe', 'HP'), ('la cie', 'LaCie'), ('aorus', 'Gigabyte'), ('hyperx', 'Kingston'), ('xpg', 'ADATA'), ('teamgroup', 'Team Group'), ('team group', 'Team Group')], key=lambda x: -len(x[0]))]

GPU_MAKERS = {'NVIDIA', 'AMD'}
SUBLINE_BRAND = {'ventus': 'MSI', 'mech': 'MSI', 'armor': 'MSI', 'gaming x': 'MSI', 'suprim': 'MSI', 'windforce': 'Gigabyte',
    'aorus': 'Gigabyte', 'eagle': 'Gigabyte', 'tuf': 'ASUS', 'strix': 'ASUS', 'rog': 'ASUS', 'phoenix': 'ASUS',
    'amp': 'ZOTAC', 'trinity': 'ZOTAC', 'twin fan': 'ZOTAC', 'ftw3': 'EVGA', 'kingpin': 'EVGA', 'xc ultra': 'EVGA',
    'nitro+': 'Sapphire', 'nitro': 'Sapphire', 'pulse': 'Sapphire', 'red dragon': 'PowerColor', 'red devil': 'PowerColor',
    'stormx': 'Palit', 'gamerock': 'Palit', 'jetstream': 'Palit', 'ichill': 'Inno3D', 'xlr8': 'PNY'}
def subline_brand(title):
    s = title.lower()
    for k, v in SUBLINE_BRAND.items():
        if re.search(r'\b' + re.escape(k) + r'(?=\W|$)', s): return v
    return None
def canon_brand(b, freq):
    k = re.sub(r'\s+', ' ', b.strip()).lower()
    if not k: return None
    if k in BRAND_ALIAS: return BRAND_ALIAS[k]
    if k in REAL_LC: return REAL_LC[k]
    # OCR-typo repair against frequent brand spellings (frequency >= 3)
    cands = [x for x, c in freq.items() if c >= 3]
    m = difflib.get_close_matches(k, [x.lower() for x in cands], n=1, cutoff=0.85)
    if m and m[0] != k:
        k = m[0]
        if k in BRAND_ALIAS: return BRAND_ALIAS[k]
        if k in REAL_LC: return REAL_LC[k]
    # fictional brands: canonical = collapse case/space
    return re.sub(r'[^a-z0-9]', '', k)

def title_brand(title):
    """Earliest-mentioned known brand; chip makers (NVIDIA/AMD) only if no board partner is named."""
    hits = []
    for p, v in TITLE_BRAND_PAT:
        m = p.search(title)
        if m: hits.append((v in GPU_MAKERS, m.start(), v))
    return min(hits)[2] if hits else None

# ---------------- capacity / vram ----------------
CAP_RE = re.compile(r'(?<![\d.,])(\d+(?:[.,]\d+)?)\s*-?\s*(tb|to|gb|go|g)(?![a-z])(?!\s*/\s*s)(?!ps)(?!\s*(?:cache|buffer|ddr|gddr|d5|d6|ram))', re.I)
def title_caps(s):
    out = []
    for m in CAP_RE.finditer(s):
        v = float(m.group(1).replace(',', '.'))
        u = m.group(2).lower()
        if u == 'g' and v < 16: continue   # '6G SAS', '12G' are link speeds
        gb = v * 1000 if u in ('tb', 'to') else v
        out.append(gb)
    return out

VRAM_RE = re.compile(r'(?<![\w.])(\d{1,2})\s*(?:gb|g)\b', re.I)

def storage_cap(row, ptype):
    title = row['title'] + ' ' + row['model']
    caps = [c for c in title_caps(title) if c >= 1]
    f = parse_num(row['storage_gb'], thousands_bias=True)
    tc = max(caps) if caps else None
    if tc is not None:
        if f is None: return tc, 'title'
        if abs(f - tc) / tc < 0.02: return tc, 'field+title'
        if abs(f * 1000 - tc) / tc < 0.02: return tc, 'title(field unit-err)'
        if f in caps: return f, 'field'
        if tc < 8 and f > tc: return f, 'field(title cap implausible)'
        return tc, 'title(conflict field=%s)' % f
    if f is not None:
        return f, 'field'
    return None, None

def gpu_vram(row):
    f = parse_num(row['vram_gb'])
    if f is not None and f > 128: f = f / 1024
    s = row['title'] + ' ' + row['model'] + ' ' + row['model_number']
    vals = [int(m.group(1)) for m in VRAM_RE.finditer(s) if 1 <= int(m.group(1)) <= 48]
    # model codes like O6G / 2GD5 / 8G
    vals += [int(x) for x in re.findall(r'(?<=[A-Za-z-])(\d{1,2})G(?:D\d|B)?(?=[\W_]|$)', s) if 1 <= int(x) <= 48]
    tv = Counter(vals).most_common(1)[0][0] if vals else None
    if f is not None and f >= 1:
        return f, 'field'
    return (float(tv), 'title') if tv else (None, None)

# ---------------- GPU chipset ----------------
def gpu_chip(s):
    s0 = s
    s = s.lower().replace('®', ' ').replace('™', ' ')
    s = re.sub(r'(?<=\d)o\b', '0', s)
    fam = None
    m = re.search(r'\b(gtx|rtx|gt|rx|gts|vega|quadro|wx|radeon pro wx|titan)\s*-?\s*(\d{2,4})\s*(ti\b|super\b|xt\b|s\b)?\s*(ti\b|super\b|xt\b)?', s)
    if m:
        fam = m.group(1).upper(); num = m.group(2)
        suf = ' '.join(x.upper().replace('SUPER', 'SUPER') for x in [m.group(3), m.group(4)] if x)
        suf = suf.replace('S', 'SUPER') if suf == 'S' else suf
        return f'{fam} {num} {suf}'.strip()
    m = re.search(r'\b(gtx|rtx|rx|gt)(\d{3,4})(ti|super|xt|s)?\b', s)
    if m:
        suf = (m.group(3) or '').upper(); suf = 'SUPER' if suf == 'S' else suf
        return f'{m.group(1).upper()} {m.group(2)} {suf}'.strip()
    m = re.search(r'\b(p|t|k|m)(\d{3,4})\b', s)
    if m and 'quadro' in s: return f'QUADRO {m.group(1).upper()}{m.group(2)}'
    m = re.search(r'\b(wx)\s?(\d{4})\b', s)
    if m: return f'WX {m.group(2)}'
    m = re.search(r'\bquadro\s+(?:volta\s+)?(gv100|gp100|rtx\s*\d{4})\b', s)
    if m: return 'QUADRO ' + m.group(1).upper().replace(' ', '')
    m = re.search(r'\bgeforce\s+(210|710|730|1030)\b', s)
    if m: return 'GT ' + m.group(1)
    m = re.search(r'\b(\d{4})\s*(ti|super)\b', s)
    if m: return f'? {m.group(1)} {m.group(2).upper()}'
    return None

# ---------------- codes / tokens ----------------
UNIT_TOK = re.compile(r'^(\d+(\.\d+)?(gb|tb|mb|go|to|g|t|m|mm|rpm|k|hz|mhz|ghz|w|bit|gbps|gb/s|mb/s|in|inch|x\d+|p|nm|pcs|pack|x|s|e|n|th|nd|rd|st|cm|ms|kg|gbit|v)|\d+(\.\d+)?|m\.?2|usb\d(\.\d)?|pcie\d(\.\d)?|gen\d|sata\d|ddr\d|gddr\d+x?|x\d+|\d+x\d+|\d+bit|\d+k|\d+-bit|3d|4k|8k|2x\d+|\d+x|v\d)$')
def codes(*fields):
    out = set()
    for f in fields:
        for tok in re.split(r'[\s,;()\[\]|"“”″\'*+]+', f):
            tok = tok.strip('.-/:')
            if len(tok) < 4: continue
            if not (re.search(r'\d', tok) and re.search(r'[A-Za-z]', tok)): continue
            n = re.sub(r'[^A-Za-z0-9]', '', tok.split('/')[0] if len(tok.split('/')[0]) >= 5 else tok).upper()
            if len(n) < 5: continue
            if UNIT_TOK.match(tok.lower()) or UNIT_TOK.match(n.lower()): continue
            out.add(n)
    return out

STOP = set('the and with for of in to a an by new retail pack box oem bulk version edition card graphics video graphic grafikkarte drive drives disk hard internal external interne interno intern extern externe disco solid state ssd hdd sata iii ii serial usb flash memory stick pen type gen pcie pci express nvme m 2 5 inch 3 in inches cache buffer rpm gb tb mb go to up read write speed speeds mb/s gb/s warranty year years 3yr 5yr black zwart silver white blue red'.split())
def tokens(s):
    s = s.lower().replace('™', ' ').replace('®', ' ')
    return [w for w in re.findall(r'[a-z0-9]+(?:\.[0-9]+)?', s)]

# ---------------- main ----------------
def main():
    t = pd.read_csv(f'{W}/state/translated.csv', dtype=str, keep_default_na=False)
    bfreq = Counter(re.sub(r'\s+', ' ', b.strip()).lower() for b in t.brand if b.strip())
    rows = []
    for _, r in t.iterrows():
        d = {}
        pt = norm_pt(r['product_type'])
        pt = OTHER_FIX.get(pt, pt) if pt else pt
        ipt = infer_pt(r['title'])
        d['n_product_type_raw'] = pt
        if pt in ('STORAGE', 'DIGITAL', 'UNCHANGED'): pt = None   # uninformative category values
        d['n_product_type'] = pt if pt else ipt
        d['f_ptype_src'] = 'field' if pt else ('title' if ipt else None)
        cb = canon_brand(r['brand'], bfreq)
        tb = title_brand(r['title'])
        d['n_brand_field'] = cb
        # matching brand: a recognised real brand in title wins over a seller/garbled brand field
        if tb and (cb is None or cb not in REAL_BRANDS or cb in GPU_MAKERS):
            d['n_brand'] = tb
        else:
            d['n_brand'] = cb if cb else tb
        if ipt == 'GPU' or pt == 'GPU':
            if d['n_brand'] in (None, 'NVIDIA', 'AMD') or d['n_brand'] not in REAL_BRANDS:
                sb = subline_brand(r['title'] + ' ' + r['model'])
                if sb: d['n_brand'] = sb
        d['n_real_brand'] = (d['n_brand'] in REAL_BRANDS)
        d['n_price'] = parse_num(r['price'])
        cur = re.sub(r'\s+', '', r['priceCurrency']).upper()
        d['n_currency'] = cur if re.fullmatch(r'[A-Z]{3}', cur) else None
        for c in ['read_speed_mb_s', 'write_speed_mb_s']:
            d['n_' + c] = parse_num(r[c], thousands_bias=True)
        for c in ['width_mm', 'length_mm', 'height_mm', 'weight_g']:
            d['n_' + c] = parse_num(r[c])
        ptype = d['n_product_type']
        if ptype != 'GPU':
            cap, how = storage_cap(r, ptype)
        else:
            cap, how = (None, None)
        d['n_cap_gb'] = cap; d['f_cap_src'] = how
        d['n_storage_gb_field'] = parse_num(r['storage_gb'], thousands_bias=True)
        if ptype == 'GPU':
            v, how = gpu_vram(r)
            d['n_vram_gb'] = v; d['f_vram_src'] = how
            ch = gpu_chip(' '.join([r['title'], r['model']])) or gpu_chip(r['chipset_name']) or gpu_chip(r['description'][:300])
            # 'SUPER' written apart from the chip number (e.g. 'RTX 2060 8GB GDDR6 SUPER MINI')
            if ch and re.match(r'(GTX 16|RTX 20)\d\d$', ch) and re.search(r'\bsuper\b', (r['title'] + ' ' + r['model']).lower()):
                ch += ' SUPER'
            d['n_chip'] = ch
        else:
            d['n_vram_gb'] = parse_num(r['vram_gb']); d['f_vram_src'] = 'field' if d['n_vram_gb'] else None
            d['n_chip'] = None
        d['n_codes'] = sorted(codes(r['title'], r['model'], r['model_number']))
        mn0 = r['model_number'].split('/')[0] if len(r['model_number'].split('/')[0]) >= 5 else r['model_number']
        d['n_mn'] = re.sub(r'[^A-Za-z0-9]', '', mn0).upper() or None
        d['n_head'] = (re.findall(r'[a-z0-9]+', r['title'].lower()) or [''])[0]
        d['n_tokens'] = tokens(r['title'] + ' ' + r['model'])
        rows.append(d)
    n = pd.concat([t, pd.DataFrame(rows)], axis=1)
    n.to_pickle(f'{W}/state/normalized.pkl')
    n.drop(columns=['description']).to_csv(f'{W}/state/normalized.csv', index=False)

    # ---- diagnostics ----
    diag = {}
    for c in ['price', 'read_speed_mb_s', 'write_speed_mb_s', 'width_mm', 'length_mm', 'height_mm', 'weight_g']:
        raw = (t[c] != '').sum(); ok = n['n_' + c].notna().sum()
        diag[c + '_parse_rate'] = round(ok / raw, 4) if raw else None
    diag['ptype_counts'] = n.n_product_type.value_counts(dropna=False).head(8).to_dict()
    diag['ptype_missing_after_infer'] = int(n.n_product_type.isna().sum())
    diag['cap_known_storage'] = float(n[n.n_product_type.isin(['SSD', 'HDD', 'USB_STICK'])].n_cap_gb.notna().mean())
    diag['vram_known_gpu'] = float(n[n.n_product_type == 'GPU'].n_vram_gb.notna().mean())
    diag['chip_known_gpu'] = float(n[n.n_product_type == 'GPU'].n_chip.notna().mean())
    print(json.dumps(diag, indent=1, default=str))
    json.dump(diag, open(f'{W}/state/norm_diag.json', 'w'), default=str, indent=1)


if __name__ == '__main__':
    main()
