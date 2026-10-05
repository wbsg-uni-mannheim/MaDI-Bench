"""Shared pairwise evidence used by matching (s4) and constrained clustering (s5)."""
import re, difflib
import pandas as pd

STORAGE = {'SSD', 'HDD', 'USB_STICK'}
UNIT_STRIP = re.compile(r'\d+(?:[.,]\d+)?\s*-?\s*(?:tb|gb|go|to|mb|mhz|ghz|rpm|bit|bits|mm|hz|w|k|gbps|gb/s|mb/s|gbit/s|gbit|in|inch|pack|pcs|years?|yr|x|g)\b', re.I)
SPEC_TOK = re.compile(r'^(gen[\d.x]*|pcie[\d.]*|usb[\d.]*|sata[\d.]*|\d+yrs?|\d+y|\d+x\d+|x[1248]|m\.?2|ddr\d+x?|gddr\d+x?|d[56]x?|m2|v\d|3d|4k|8k|\d{1,2}|2280|2260|2242|7200|5400|10000|15000|512[en]|4kn|2\.5|3\.5|1\.8|3\.0|3\.1|3\.2|2\.0|4\.0|\d+nm|nvme\d*|hbm\d*e?|type\d*|mu|hdmi\d*|dp\d*|dvi\d*|a\d|b\d|c\d)$')
VARIANT_WORDS = ['evo', 'pro', 'qvo', 'plus', 'touch', 'max', 'ultra', 'lite', 'extreme', 'blade', 'glide', 'fit', 'dual', 'luxe']
STORAGE_LINES = ['barracuda', 'ironwolf', 'skyhawk', 'exos', 'firecuda', 'constellation', 'cheetah', 'savvio',
                 'blue', 'black', 'red', 'purple', 'green', 'gold', 'passport', 'elements', 'mybook', 'backup',
                 'expansion', 'rugged', 'savage', 'fury', 'triple', 'hub', 'slim', 'xbox', 'ps4', 'kc2000', 'a400', 'a2000', 'uv500', 'mx500', 'bx500', 'p300', 'n300', 'x300', 'l200']
GPU_SUBLINES = ['tuf', 'phoenix', 'strix', 'rog', 'ventus', 'armor', 'mech', 'windforce', 'eagle', 'aorus', 'amp', 'twin',
                'mini', 'trinity', 'xc', 'ftw3', 'ko', 'turbo', 'pulse', 'nitro', 'dragon', 'devil', 'evoke', 'jetstream',
                'gamerock', 'stormx', 'xlr8', 'ichill', 'challenger', 'fighter', 'vision', 'gaming', 'passive', 'silent',
                'lp', 'low', 'blower', 'founders', 'trio', 'xtreme', 'waterforce', 'aero', 'ventus', 'xs', 'dual', 'ko', 'sc', 'ultra', 'hybrid', 'kingpin', 'suprim', 'aero', 'master', 'elite', 'itx']

WD_COLORS = {'blue', 'black', 'red', 'purple', 'green', 'gold'}
USB_LINES = {'blade', 'glide', 'force', 'ultra', 'fit', 'flair', 'luxe', 'dual', 'ixpand', 'extreme', 'cruzer', 'switch',
             'traveler', 'traveller', 'datatraveler', 'microduo', 'duo', 'locker', 'vault', 'exodia', 'kyson', 'canvas'}
USB_LINES -= {'cruzer', 'traveler', 'traveller', 'datatraveler'}

def title_words(r):
    s = (r['title'] + ' ' + r['model'] + ' ' + r['model_number']).lower().replace('™', ' ').replace('®', ' ')
    s = re.sub(r'\bdual[- ]?(slot|fan|fans|bios|hdmi|dvi|display)', ' ', s)
    return set(re.findall(r'[a-z]+', s))

def model_tokens(r):
    """Digit-bearing model tokens (e.g. t5, 860, sx8200, s11, 7e2000), with units/spec numbers removed."""
    s = (r['title'] + ' ' + r['model'] + ' ' + r['model_number']).lower().replace('™', ' ').replace('®', ' ')
    s = re.sub(r'\b[rw]?:?\d{2,5}\s*/\s*[rw]?:?\d{2,5}\b', ' ', s)   # read/write speed pairs '3500/1200'
    s = UNIT_STRIP.sub(' ', s)
    # drop hyphen/slash-joined part numbers (>=8 alnum chars): their fragments are not model names
    s = ' '.join(t for t in re.split(r'[\s,;()\[\]|]+', s)
                 if not (re.search(r'[-/]', t) and len(re.sub(r'[^a-z0-9]', '', t)) >= 8 and re.search(r'\d', t)))
    s = re.sub(r'([a-z]{4,})(\d)', r'\1 \2', s)   # 'datatraveler100' -> 'datatraveler 100'
    out = set()
    for tok in re.split(r'[^a-z0-9.]+', s):
        tok = tok.strip('.')
        if not tok or not re.search(r'\d', tok): continue
        if len(tok) >= 8: continue            # long part numbers are handled as codes
        if SPEC_TOK.match(tok): continue
        if tok == 'x16' and r['n_product_type'] != 'HDD': continue
        if re.fullmatch(r'\d+\.\d+', tok): continue
        out.add(tok)
    return out

def near(a, b):
    """Same code up to OCR noise (letter<->digit confusions), a regional/colour letter suffix, or an appended suffix."""
    if a == b: return True
    if min(len(a), len(b)) < 6: return False
    if len(a) == len(b):
        mm = [(x, y) for x, y in zip(a, b) if x != y]
        if len(mm) <= 2 and all(x.isdigit() != y.isdigit() for x, y in mm): return True
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    if k >= 8 and a[k:].isalpha() and b[k:].isalpha() and 2 <= max(len(a) - k, len(b) - k) <= 3: return True
    if k >= 8 and (k == len(a) or k == len(b)): return True
    return False

def rel_diff(a, b):
    m = max(a, b)
    return abs(a - b) / m if m else 0.0

def hard_conflict(ra, rb):
    """Contradictions of identity-defining specs; missing values never conflict."""
    pa, pb = ra['n_product_type'], rb['n_product_type']
    INS = {'GPU', 'SSD', 'HDD', 'USB_STICK'}
    if pa in INS and pb in INS and pa != pb: return 'ptype'
    if pa and pb and pa != pb and not (set(pa.split('_')) & set(pb.split('_'))) \
            and difflib.SequenceMatcher(None, pa, pb).ratio() < 0.6:
        return 'ptype'   # out-of-scope categories: tolerate noisy spellings/partial names, not unrelated types
    if pd.notna(ra['n_cap_gb']) and pd.notna(rb['n_cap_gb']) and rel_diff(ra['n_cap_gb'], rb['n_cap_gb']) > 0.02 \
            and (pa in STORAGE or pb in STORAGE):
        return 'capacity'
    if pa == 'GPU' or pb == 'GPU':
        if pd.notna(ra['n_vram_gb']) and pd.notna(rb['n_vram_gb']) and rel_diff(ra['n_vram_gb'], rb['n_vram_gb']) > 0.02:
            return 'vram'
        ca, cb = ra['n_chip'], rb['n_chip']
        if ca and cb and ca != cb and not (ca.startswith('?') or cb.startswith('?')): return 'chip'
    if not ra['n_real_brand'] and not rb['n_real_brand']:
        # fictional/unknown brands: brand fields are noisy (suffixes, reordering, OCR) -> only clearly different ones conflict
        fa_, fb_ = ra['n_brand_field'], rb['n_brand_field']
        if fa_ and fb_:
            if fa_ not in fb_ and fb_ not in fa_ and sorted(fa_) != sorted(fb_) \
                    and difflib.SequenceMatcher(None, fa_, fb_).ratio() < 0.6: return 'brand_head'
        else:
            ha, hb = ra['n_head'], rb['n_head']
            if ha and hb and ha != hb and difflib.SequenceMatcher(None, ha, hb).ratio() < 0.6: return 'brand_head'
    if pa == 'GPU' or pb == 'GPU':
        ma, mb = gpu_mem(ra), gpu_mem(rb)
        if ma and mb and ma != mb: return 'gpu_memtype'
    return None

def gpu_mem(r):
    s = (r['title'] + ' ' + r['model'] + ' ' + r['memory_type']).lower()
    m = re.findall(r'\b(g?ddr\s?[3-6]x?|hbm2?e?)\b', s) + ['gddr' + x for x in re.findall(r'\bd([56])\b', s)]
    m = {x.replace(' ', '') for x in m}
    # on GPUs, 'DDR5'/'DDR6' are shorthand for GDDR5/GDDR6
    m = {('g' + x if x.startswith('ddr') and x[3] in '56' else x) for x in m}
    return frozenset(m) if len(m) == 1 else None

def gpu_quad(r):
    """True for 4-HDMI boards (ASUS GT710-4H); False when an explicit board code without 4H is given; else unknown."""
    s = (r['title'] + ' ' + r['model'] + ' ' + r['model_number']).lower()
    if re.search(r'-4h-|quad hdmi|4\s*\*?\s*x?\s*hdmi|4 hdmi', s): return True
    if re.search(r'gt710-sl-2gd5|gt710-2gd5', s): return False
    return None

def gpu_edition(r):
    """OC ('o') / advanced ('a') / plain ('') edition from hyphenated board codes (e.g. -O6G-, -A11G-, -6G-) or an 'OC' word."""
    s = (r['title'] + ' ' + r['model'] + ' ' + r['model_number']).upper()
    eds = set(re.findall(r'-(O|A)?\d{1,2}G(?:D\d)?(?=-|\b)', s)) | set(re.findall(r'\b(O|A)\d{1,2}G\b', s))
    if re.search(r'\bOC\b', s): eds.add('O')
    return eds.pop() if len(eds) == 1 else None

def tok_overlap(ma, mb):
    ja, jb = ' '.join(ma), ' '.join(mb)
    return bool(ma & mb) or any(len(t) >= 3 and t in jb for t in ma) or any(len(t) >= 3 and t in ja for t in mb)

def soft_conflict(ra, rb, fa, fb):
    """Contradictions overridden only by a shared long part code (brand: OEM relabels share part numbers)."""
    if (ra['n_real_brand'] or rb['n_real_brand']) and ra['n_brand'] and rb['n_brand'] and ra['n_brand'] != rb['n_brand'] \
            and ra['n_brand'] not in ('NVIDIA', 'AMD') and rb['n_brand'] not in ('NVIDIA', 'AMD'):
        return 'brand'
    ma, mb = fa['mtok'], fb['mtok']
    pt0 = ra['n_product_type'] or rb['n_product_type']
    if pt0 != 'GPU' and ma and mb and not tok_overlap(ma, mb):   # GPU titles: chip/subline/edition rules instead
        return 'model_tokens'
    wa, wb = fa['words'], fb['words']
    pt0 = ra['n_product_type'] or rb['n_product_type']
    for v in (['evo', 'pro', 'qvo', 'plus', 'touch', 'heatsink'] if pt0 != 'GPU' else []):
        if (v in wa) != (v in wb):
            # a missing variant word only conflicts if the other record names a sibling variant or line is shared
            if v in ('evo', 'qvo', 'pro') and (({'evo', 'qvo', 'pro'} & wa) and ({'evo', 'qvo', 'pro'} & wb)) and \
                    ({'evo', 'qvo', 'pro'} & wa) != ({'evo', 'qvo', 'pro'} & wb):
                return 'variant:' + v
            if v == 'heatsink' and 'Western Digital' not in (ra['n_brand'], rb['n_brand']): continue
            if v in ('plus', 'touch', 'pro', 'heatsink') and ((ma & mb) or (wa & wb & set(STORAGE_LINES))):
                return 'variant:' + v
    for t in (ma ^ mb if pt0 != 'GPU' else ()):   # on GPUs 'G5'/'G6' abbreviate the memory type
        if re.fullmatch(r'g\d', t) and tok_overlap(ma, mb):
            other = ' '.join(mb if t in ma else ma)
            if t not in other: return 'generation'
    pt = ra['n_product_type'] or rb['n_product_type']
    if pt in ('SSD', 'HDD'):
        sa, sb = storage_form(ra), storage_form(rb)
        if len(sa) == 1 and len(sb) == 1 and sa != sb: return 'form_factor'
        ia, ib = storage_iface(ra), storage_iface(rb)
        if ia and ib and not (ia & ib): return 'interface'
        if pt == 'HDD':
            xa, xb = hdd_rpm(ra), hdd_rpm(rb)
            if xa and xb and xa != xb: return 'rpm'
            xa, xb = sector_fmt(ra), sector_fmt(rb)
            if xa and xb and xa != xb: return 'sector_format'
            xa, xb = sas_speed(ra), sas_speed(rb)
            if xa and xb and xa != xb: return 'sas_speed'
    if pt in STORAGE:
        LINES = set(STORAGE_LINES) - (set() if 'Western Digital' in (ra['n_brand'], rb['n_brand']) else WD_COLORS)
        la = (wa & (USB_LINES if pt == 'USB_STICK' else LINES)) | {t for t in ma if not re.fullmatch(r'g\d', t)}
        lb = (wb & (USB_LINES if pt == 'USB_STICK' else LINES)) | {t for t in mb if not re.fullmatch(r'g\d', t)}
        if la and lb and not tok_overlap(la, lb): return 'line_or_model'
        if pt == 'USB_STICK':
            for v in ('flair', 'luxe', 'fit', 'dual', 'go', 'plus', 'locker', 'vault', 'duo', 'microduo'):
                if (v in wa) != (v in wb) and tok_overlap(la, lb): return 'usb_variant:' + v
    if pt == 'USB_STICK':
        la, lb = wa & USB_LINES, wb & USB_LINES
        if la and lb and not (la & lb): return 'usb_line'
    if pt in STORAGE:
        LINES = set(STORAGE_LINES) - (set() if 'Western Digital' in (ra['n_brand'], rb['n_brand']) else WD_COLORS)
        la, lb = wa & LINES, wb & LINES
        if la and lb and not (la & lb): return 'line'
    if pt == 'GPU':
        la, lb = wa & set(GPU_SUBLINES), wb & set(GPU_SUBLINES)
        if ra['n_brand'] not in ('Gigabyte', 'MSI'):
            la -= {'gaming'}; lb -= {'gaming'}
        if ra['n_brand'] != 'ASUS':
            la -= {'dual'}; lb -= {'dual'}
        if la and lb and not (la & lb): return 'gpu_subline'
        if ra['n_brand'] == 'Gigabyte' and rb['n_brand'] == 'Gigabyte' and (('gaming' in wa) != ('gaming' in wb)):
            other = wb if 'gaming' in wa else wa
            if 'oc' in other or (other & set(GPU_SUBLINES) - {'gaming'}): return 'gpu_gigabyte_gaming'
        qa, qb = gpu_quad(ra), gpu_quad(rb)
        if qa is not None and qb is not None and qa != qb: return 'gpu_quad_hdmi'
        ea, eb = gpu_edition(ra), gpu_edition(rb)
        if ea is not None and eb is not None and ea != eb: return 'gpu_edition'
    return None

def strong_shared(ca, cb):
    """Shared long (>=8 char) part-number-like codes, allowing OCR/suffix noise."""
    return {x for x in ca for y in cb if len(x) >= 8 and len(y) >= 8 and near(x, y)}

def shape(x):
    return re.sub(r'[A-Z]', 'L', re.sub(r'\d', 'D', x))

def partno_conflict(ca, cb):
    """Two same-family part numbers (same 3-char prefix or same letter/digit shape) that are not near-equal."""
    if strong_shared(ca, cb): return False
    A = [x for x in ca if len(x) >= 6]; B = [y for y in cb if len(y) >= 6]
    for x in A:
        for y in B:
            if near(x, y): continue
            if (x[:3] == y[:3] and abs(len(x) - len(y)) <= 3) or shape(x) == shape(y):
                return True
    return False

def storage_form(r):
    s = (r['title'] + ' ' + r['model']).lower()
    f = set()
    if re.search(r'\bm\.?2\b|\b22(80|42|60)\b', s): f.add('m2')
    if re.search(r'\b2[.,]5\s*(\"|\'|”|″|-?inch|in\b|zoll)|\b2[.,]5\b(?!\s*gb)|\bsff\b', s): f.add('2.5')
    if re.search(r'\b3[.,]5\s*(\"|\'|”|″|-?inch|in\b|zoll)|\b3[.,]5\b(?!\s*gb)|\blff\b', s): f.add('3.5')
    if re.search(r'\bmsata\b', s): f.add('msata')
    return f

def storage_iface(r):
    s = (r['title'] + ' ' + r['model']).lower()
    f = set()
    if re.search(r'\bsas\b|serial attached', s): f.add('sas')
    if re.search(r'\bs-?ata|serial ata', s): f.add('sata')
    if re.search(r'\bnvme\b|\bpcie\b|pci[- ]?e|pci express', s): f.add('nvme')
    if re.search(r'\busb|\bexternal|\bextern|\bportable|thunderbolt|firewire', s): f.add('usb')
    return f

def hdd_rpm(r):
    s = (r['title'] + ' ' + r['model']).lower()
    m = re.findall(r'\b(5400|5900|7200|10000|10k|10\.000|15000|15k|15\.000|7\.2k)\s*(?:rpm)?\b', s)
    m = {x.replace('.', '').replace('7.2k', '7200').replace('72k', '7200').replace('10k', '10000').replace('15k', '15000') for x in m}
    return m.pop() if len(m) == 1 else None

def sector_fmt(r):
    m = set(re.findall(r'\b512\s?([en])\b', (r['title'] + ' ' + r['model']).lower()))
    return m.pop() if len(m) == 1 else None

def sas_speed(r):
    s = (r['title'] + ' ' + r['model']).lower()
    m = set(re.findall(r'\b(3|6|12)\s*g(?:b(?:ps|/s|it/s)?)?\b', s))
    return m.pop() if len(m) == 1 and 'sas' in s else None


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def mn_noise_equal(a, b):
    """Model numbers equal up to OCR-style substitutions (<=2, same length) or a <=2-char appended suffix."""
    if a == b: return True
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) <= 2
    s_, l_ = sorted((a, b), key=len)
    return l_.startswith(s_) and ((len(l_) - len(s_) <= 2 and len(s_) >= 5) or len(s_) >= 8)


def key_agreement(ra, rb, fa, fb):
    """Positive identity evidence beyond title similarity: same real brand, same capacity (storage) and a shared
    distinctive model token; for GPUs same chip + VRAM and a shared board sub-line word."""
    if not (ra['n_real_brand'] and rb['n_real_brand'] and ra['n_brand'] == rb['n_brand']): return False
    pt = ra['n_product_type'] or rb['n_product_type']
    if pt == 'GPU':
        if not (ra['n_chip'] and ra['n_chip'] == rb['n_chip']): return False
        if pd.isna(ra['n_vram_gb']) or pd.isna(rb['n_vram_gb']) or ra['n_vram_gb'] != rb['n_vram_gb']: return False
        subl = set(GPU_SUBLINES) - {'gaming', 'low', 'lp', 'dual'}
        return bool(fa['words'] & fb['words'] & subl)
    if pd.isna(ra['n_cap_gb']) or pd.isna(rb['n_cap_gb']) or rel_diff(ra['n_cap_gb'], rb['n_cap_gb']) > 0.02: return False
    shared = {t for t in fa['mtok'] & fb['mtok'] if len(t) >= 3 and not re.fullmatch(r'g\d', t)}
    return bool(shared)
