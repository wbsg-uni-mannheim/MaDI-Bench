"""Stage 2: normalization. Produces work/state/norm.pkl (raw values kept as raw_<col>)."""
import json, re, os, collections
import pandas as pd, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
mapping = json.load(open('work/state/sm_mapping.json'))
frames = []
for src, m in mapping.items():
    df = pd.read_csv(f'task/input/data/{src}.csv', dtype=str, keep_default_na=False, na_values=[''])
    df = df.rename(columns=m)
    df.insert(0, 'source', src)
    frames.append(df)
d = pd.concat(frames, ignore_index=True)
for c in d.columns:
    if d[c].dtype == object:
        d[c] = d[c].str.strip().replace({'': np.nan})
for c in list(d.columns):
    if c not in ('source', 'id'):
        d['raw_' + c] = d[c]

# ---------- numeric parsing (remove thousands separators and injected spaces) ----------
NUM = ['price', 'vram_gb', 'storage_gb', 'read_speed_mb_s', 'write_speed_mb_s', 'width_mm', 'length_mm', 'height_mm', 'weight_g']
parse_fail = {}
def parse_price(v):
    if pd.isna(v): return np.nan
    t = re.sub(r'[^0-9.,]', '', re.sub(r'(?<=\d)\s+(?=\d)', '', v))
    t = t.strip('.,')
    if not t: return np.nan
    if ',' in t and '.' in t:
        dec = ',' if t.rfind(',') > t.rfind('.') else '.'
        t = t.replace('.' if dec == ',' else ',', '').replace(',', '.')
    elif ',' in t:
        t = t.replace(',', '') if re.search(r',\d{3}$', t) else t.replace(',', '.')
    try: return float(t)
    except ValueError: return np.nan
price_raw = d['price'].copy()
d['price'] = d['price'].map(parse_price)
parse_fail['price'] = int((price_raw.notna() & d['price'].isna()).sum())
for c in NUM[1:]:
    s = d[c].str.replace(r'[\s,]', '', regex=True)
    v = pd.to_numeric(s, errors='coerce')
    parse_fail[c] = int((d[c].notna() & v.isna()).sum())
    d[c] = v.round(4)
# VRAM reported in MB (4096, 6144, 8192) -> GB
d.loc[d.vram_gb >= 512, 'vram_gb'] = (d.loc[d.vram_gb >= 512, 'vram_gb'] / 1024).round(2)
# zero is not a valid value for these attributes (schema minimums >= 0.1/1)
for c in NUM[1:]:
    d.loc[d[c] <= 0, c] = np.nan

# ---------- capacity from title ----------
def title_caps(t):
    out = set()
    for a, u in re.findall(r'(\d+(?:[.,]\d+)?)\s*(TB|GB|GO|TO)(?![A-Za-z/])(?!\s*/\s*s)', t or '', re.I):
        x = float(a.replace(',', '.')) * (1000 if u.upper() in ('TB', 'TO') else 1)
        out.add(round(x, 1))
    return out
d['title_caps'] = d.title.map(title_caps)
# storage_gb given in TB (e.g. 4.0 for a '4TB' title) -> convert to GB, only when the title confirms it
conv = 0
for i, r in d.iterrows():
    sg = r.storage_gb
    if pd.notna(sg) and r.title_caps and sg < 100:
        if any(abs(c - sg * 1000) / (sg * 1000) < 0.03 for c in r.title_caps) and not any(abs(c - sg) / sg < 0.03 for c in r.title_caps):
            d.at[i, 'storage_gb'] = sg * 1000; conv += 1
def cap_key(r):
    # capacity used for matching: unique title capacity if unambiguous, otherwise storage_gb
    tc = r.title_caps
    if r.product_type_c in ('GPU', 'RAM'):
        return np.nan
    if len(tc) == 1:
        return next(iter(tc))
    if len(tc) == 0 and pd.notna(r.storage_gb):
        return r.storage_gb
    return np.nan

# ---------- product type ----------
PT_ALIAS = {'USB FLASH DRIVE': 'USB_STICK', 'FLASH DRIVE': 'USB_STICK', 'EXTERNAL_SSD': 'SSD'}
def ptype(v):
    if pd.isna(v): return np.nan
    k = re.sub(r'[\s\-]+', '_', v.strip().upper())
    return PT_ALIAS.get(k.replace('_', ' '), PT_ALIAS.get(k, k))
d['product_type_c'] = d.product_type.map(ptype)
d['cap_key'] = d.apply(cap_key, axis=1)

# ---------- brand ----------
BRAND_ALIAS = {'wd': 'westerndigital', 'adata': 'adata', 'kingstontechnology': 'kingston', 'hpe': 'hp', 'hewlettpackard': 'hp',
               'hewlettpackardenterprise': 'hp', 'lacie': 'lacie', 'pnytechnologies': 'pny', 'samsungelectronics': 'samsung',
               'patriotmemory': 'patriot', 'seagatetechnology': 'seagate', 'zotacgaming': 'zotac', 'teamgroup': 'team', 'thinkpad': 'lenovo', 'teamgroupinc': 'team', 'gigabytetechnology': 'gigabyte', 'aorus': 'gigabyte', 'toshibaelectronics': 'toshiba'}
def bkey(v):
    if pd.isna(v): return np.nan
    k = re.sub(r'[^a-z0-9]', '', v.lower())
    if k.startswith('wd') or k.startswith('westerndigital'): k = 'westerndigital'
    return BRAND_ALIAS.get(k, k) or np.nan
d['brand_key'] = d.brand.map(bkey)
# brands mentioned in the title (vocabulary = brand keys seen >= 3 times in the brand column)
bc = d.brand_key.value_counts()
vocab = set(bc[bc >= 3].index) | {'westerndigital', 'wd', 'crucial', 'nvidia', 'amd'}
vocab -= {'nvidia', 'amd', 'intel'}   # chip vendors appear in partner-card titles
def title_brands(t):
    toks = re.findall(r'[a-z0-9]+', (t or '').lower())
    found = set()
    for n in (1, 2):
        for j in range(len(toks) - n + 1):
            k = ''.join(toks[j:j + n]); k = BRAND_ALIAS.get(k, k)
            if k in vocab: found.add(k)
    return found
d['title_brands'] = d.title.map(title_brands)

# ---------- identifier codes ----------
UNIT = re.compile(r'^\d+(\.\d+)?(GB|TB|MB|GO|TO|MHZ|GHZ|RPM|MM|W|HZ|GBPS|GBS|MBS|K|KRPM|G|X|M|IN|INCH|NM|BIT|TBW|IOPS|P|D|FPS)?$')
def norm_code(t):
    return re.sub(r'[^A-Z0-9]', '', t.upper())
def codes(*texts):
    out = set()
    for t in texts:
        if pd.isna(t): continue
        for tok in re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_./+]*[A-Za-z0-9]', t):
            cands = {tok} | set(re.split(r'[/_]', tok))
            for c in cands:
                n = norm_code(c)
                if len(n) >= 5 and re.search(r'\d', n) and re.search(r'[A-Z]', n) and not UNIT.match(n) \
                        and not re.match(r'^(USB|PCIE|SATA|DDR|GDDR|LPDDR|NVME|GEN)\d', n) and not re.match(r'^\d+(GB|TB|MB)', n):
                    out.add(n)
    return out
d['mn_key'] = d.model_number.map(lambda v: norm_code(v) if pd.notna(v) and len(norm_code(v)) >= 3 else np.nan)
d['codes'] = [codes(r.model_number, r.model, r.title) for r in d.itertuples()]
d['title_codes'] = [codes(r.title) for r in d.itertuples()]

# ---------- GPU chip designation from title/model/chipset ----------
def gpu_chip(r):
    txt = ' '.join(str(x) for x in (r.title, r.model, r.chipset_name) if pd.notna(x)).upper()
    m = re.search(r'\b(GTX|RTX|GT|RX|R9|R7|QUADRO\s*[A-Z]?|TITAN)\s*-?(\d{3,4})\s*(TI|SUPER|XT|S)?\b', txt)
    if not m: return np.nan
    suf = {'S': 'SUPER'}.get(m.group(3), m.group(3)) or ''
    if not suf and m.group(1) in ('GTX', 'RTX') and re.search(r'\bSUPER\b', txt): suf = 'SUPER'
    return f'{m.group(1).replace(" ", "")}{m.group(2)}{suf}'
d['gpu_chip'] = d.apply(gpu_chip, axis=1)
d.loc[d.product_type_c != 'GPU', 'gpu_chip'] = np.nan

d['title_n'] = (d.title.fillna('').str.lower().str.replace(r'(?<=[a-z0-9])\+(?=\s|$|[^a-z0-9])', ' plus ', regex=True)
                .str.replace(r'[^a-z0-9.]+', ' ', regex=True).str.replace(r'(?<![0-9])\.|\.(?![0-9])', ' ', regex=True)
                .str.replace(r'(?<=\d) (gb|tb|mb|go|to|g|k|rpm|mhz|gbps|bit|hz|w|mm)\b', r'\1', regex=True)
                .str.replace(r'\s+', ' ', regex=True).str.strip())
# GPU memory size stated in the title (e.g. '8GB', '8G', '6144 MB'); used as matching evidence for GPUs
def title_vram(r):
    if r.product_type_c != 'GPU': return np.nan
    t = str(r.title)
    v = {float(x) for x in re.findall(r'(\d{1,2})\s*G(?:B)?\b(?![/])', t, re.I)} | \
        {round(float(x) / 1024) for x in re.findall(r'\b(\d{4,5})\s*MB\b', t, re.I)}
    v = {x for x in v if 1 <= x <= 48}
    return next(iter(v)) if len(v) == 1 else np.nan
d['vram_key'] = d.apply(title_vram, axis=1)
d['vram_key'] = d.vram_key.fillna(d.vram_gb.where(d.product_type_c == 'GPU'))
# spindle speed stated in the title (7.2K / 7200 RPM / 10K)
def rpm(t):
    t = str(t).upper()
    v = {float(x) * 1000 for x in re.findall(r'\b(\d{1,2}(?:\.\d)?)\s*K(?:\s*RPM)?\b', t)} | {float(x) for x in re.findall(r'\b(\d{4,5})\s*RPM', t)}
    v = {x for x in v if 4000 <= x <= 15000}
    return next(iter(v)) if len(v) == 1 else np.nan
d['rpm_key'] = d.title.map(rpm)
d.loc[~d.product_type_c.isin(['HDD']) & d.product_type_c.notna(), 'rpm_key'] = np.nan
d.to_pickle('work/state/norm.pkl')
stats = {'rows': len(d), 'numeric_parse_fail': parse_fail, 'storage_tb_to_gb_conversions': conv,
         'cap_key_density': float(d.cap_key.notna().mean()), 'mn_key_density': float(d.mn_key.notna().mean()),
         'title_brand_density': float((d.title_brands.map(len) > 0).mean()), 'vram_key_on_gpu': float(d[d.product_type_c == 'GPU'].vram_key.notna().mean()), 'rpm_key_on_hdd': float(d[d.product_type_c == 'HDD'].rpm_key.notna().mean()), 'gpu_chip_on_gpu': float(d[d.product_type_c == 'GPU'].gpu_chip.notna().mean())}
print(json.dumps(stats))
json.dump(stats, open('work/state/norm_stats.json', 'w'))
