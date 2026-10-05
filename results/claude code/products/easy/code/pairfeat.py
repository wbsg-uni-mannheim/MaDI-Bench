"""Shared pairwise evidence functions (used by s4 matching and s5 clustering)."""
import re
from difflib import SequenceMatcher
import pandas as pd
CHIP_VENDORS = {'nvidia', 'amd'}
MAX_CODE_DF = 16
def num_conf(a, b, tol):
    return pd.notna(a) and pd.notna(b) and abs(a - b) / max(a, b) > tol
def mn_conflict(x, y, codes_x=(), codes_y=(), allow_scheme=False):
    # both have a model number, they differ, neither contains the other (e.g. SEDC450R3840G vs DC450R3840G),
    # and neither record mentions the other's model number elsewhere (vendors list several part numbers)
    if pd.isna(x) or pd.isna(y) or x == y: return False
    if min(len(x), len(y)) >= 4 and (x in y or y in x): return False
    if x in codes_y or y in codes_x: return False
    # a mostly-digit vendor part number (ASUS '90YV0DK3M0NA00') vs a model-name code ('DUALRTX2070S8GEVO') are two numbering
    # schemes for one card and cannot contradict each other; applied to the four hardware types only (checked by caller)
    dr = lambda z: sum(ch.isdigit() for ch in z) / len(z)
    if allow_scheme and (dr(x) >= 0.6) != (dr(y) >= 0.6) and x[:3] != y[:3] and SequenceMatcher(None, x, y).ratio() < 0.6:
        return False
    return True
# Model modifiers: present on one side only -> different variant (e.g. 970 EVO vs 970 EVO Plus, 2080 AMP vs AMP Extreme)
MODIFIERS = set('pro plus super ti xt extreme ultra max lite mini slim hub go touch elite advanced'.split())
# Product lines / colours: conflict only when both sides name one and they are disjoint (omission is unknown)
LINES = set('strix tuf dual phoenix turbo aorus windforce eagle ventus trio aero mech amp twin evo qvo gamingpro thicc pulse nitro '
            'red green blue black purple gold silver grey gray white pink amber orange yellow'.split())
def _present(t, toks):
    return any(t == u or (len(t) >= 3 and len(u) > len(t) and (u.startswith(t) or u.endswith(t))) for u in toks)
def variant_diff(ta, tb):
    a, b = set(ta.split()), set(tb.split())
    for t in (a - b) & MODIFIERS:
        if not _present(t, b): return True
    for t in (b - a) & MODIFIERS:
        if not _present(t, a): return True
    la, lb = a & LINES, b & LINES
    return bool((la - lb) and (lb - la))
UNITISH = re.compile(r'^(\d+(\.\d+)?(gb|tb|mb|g|k|m|rpm|mhz|ghz|hz|mm|w|bit|x|p|nm|in|inch|gbps|gbit|mbs|tbw|d|s|e|n|kn|fps)|'
                     r'(usb|gen|pcie|sata|ddr|gddr|lpddr|nvme|x|m|d|r|w|hdmi|dp|type)\d.*|\d+x|x\d+|\d+(st|nd|rd|th))$')
def model_tokens(t):
    # short alphanumeric model designators (t5, p10, sn550, u202, 970) - excluding units and interface generations
    out = set()
    for u in t.split():
        if UNITISH.match(u): continue
        if re.fullmatch(r'[a-z]{1,4}\d{1,5}[a-z]{0,3}', u) or re.fullmatch(r'\d{2,4}[a-z]{0,2}', u):
            if re.fullmatch(r'\d{4}', u) and u in ('2280', '2242', '2230'): continue
            out.add(u)
    return out
def same_code(x, y):
    return x == y or (min(len(x), len(y)) >= 5 and (x.startswith(y) or y.startswith(x) or x in y or y in x))
IFACE = {'sas', 'sata', 'nvme'}
HW = {'GPU', 'SSD', 'HDD', 'USB_STICK'}
WORD_DF, WORD_BRANDS = {}, {}
def set_word_df(titles, brand_keys=None):
    WORD_DF.clear(); WORD_BRANDS.clear()
    for i, t in enumerate(titles):
        for w in set(t.split()):
            WORD_DF[w] = WORD_DF.get(w, 0) + 1
            if brand_keys is not None and isinstance(brand_keys[i], str):
                WORD_BRANDS.setdefault(w, set()).add(brand_keys[i])
def side_specific(ta, tb, brand_vocab=()):
    # words of ta absent from tb that are brand-specific (used with a single brand in the corpus) or model tokens
    a, b = ta.split(), set(tb.split())
    out = {w for w in set(a) if w.isalpha() and len(w) >= 3 and len(WORD_BRANDS.get(w, ())) <= 1 and w not in brand_vocab
           and not _present(w, b)}
    out |= {m for m in model_tokens(ta) if not _present(m, b) and not any(m.endswith(x) or x.endswith(m) for x in model_tokens(tb))}
    return out
def drive_ff(t):
    t = str(t).lower()
    out = set()
    if re.search(r'2[.,]5\s*\\?(\"|\'|”|″|-?\s*inch|in\b|zoll)|\bsff\b', t): out.add('2.5')
    if re.search(r'3[.,]5\s*\\?(\"|\'|”|″|-?\s*inch|in\b|zoll)|\blff\b', t): out.add('3.5')
    return out
def brands(r):
    return r['title_brands'] or ({r['brand_key']} - CHIP_VENDORS if pd.notna(r['brand_key']) else set())
def pair_features(ra, rb, code_df):
    mn_eq = pd.notna(ra['mn_key']) and ra['mn_key'] == rb['mn_key']
    shared = {x for x in ra['codes'] & rb['codes'] if code_df[x] <= MAX_CODE_DF}
    # title codes on both sides that are disjoint = contradicting part numbers
    ta, tb = ra['title_codes'], rb['title_codes']
    code_contra = bool(ta and tb) and not any(same_code(x, y) for x in ta | ra['codes'] for y in tb | rb['codes'])
    # text-level contradictions used by the title-only rule
    ma, mb = model_tokens(ra['title_n']), model_tokens(rb['title_n'])
    mtok_contra = bool(ma and mb) and not any(x == y or (min(len(x), len(y)) >= 3 and (x.endswith(y) or y.endswith(x))) for x in ma for y in mb)
    ia, ib = set(ra['title_n'].split()) & IFACE, set(rb['title_n'].split()) & IFACE
    iface_contra = bool(ia and ib and not (ia & ib))
    fa, fb = drive_ff(ra['title']), drive_ff(rb['title'])
    iface_contra = iface_contra or bool(fa and fb and not (fa & fb))   # 2.5-inch vs 3.5-inch drives
    pa, pb = ra['product_type_c'], rb['product_type_c']
    type_conf = pd.notna(pa) and pd.notna(pb) and pa != pb
    cap_conf = num_conf(ra['cap_key'], rb['cap_key'], 0.03)
    vram_conf = num_conf(ra['vram_key'], rb['vram_key'], 0.03)
    rpm_conf = num_conf(ra['rpm_key'], rb['rpm_key'], 0.15)   # 5400 vs 5900 'class' disagreements are common
    chip_conf = pd.notna(ra['gpu_chip']) and pd.notna(rb['gpu_chip']) and ra['gpu_chip'] != rb['gpu_chip']
    ba, bb = brands(ra), brands(rb)
    # brand contradiction; overridden when both titles carry the same part number (OEM rebadged drives)
    brand_conf = bool(ba and bb) and not any(x.startswith(y) or y.startswith(x) for x in ba for y in bb) and not (ta & tb)
    mn_conf = mn_conflict(ra['mn_key'], rb['mn_key'], ra['codes'], rb['codes'],
                          allow_scheme=ra['product_type_c'] in HW and rb['product_type_c'] in HW)
    # positive specification agreement (both values present and equal)
    spec_agree = (pd.notna(ra['cap_key']) and pd.notna(rb['cap_key']) and not cap_conf) or \
                 (pd.notna(ra['vram_key']) and pd.notna(rb['vram_key']) and not vram_conf and pd.notna(ra['gpu_chip']) and ra['gpu_chip'] == rb['gpu_chip'])
    brand_agree = bool(ba and bb) and not brand_conf
    sa, sb = side_specific(ra['title_n'], rb['title_n'], ba | bb), side_specific(rb['title_n'], ra['title_n'], ba | bb)
    two_sided = bool(sa and sb)
    return dict(two_sided=two_sided, spec_agree=spec_agree, brand_agree=brand_agree, mn_conf=mn_conf, vdiff=variant_diff(ra['title_n'], rb['title_n']), mn_eq=mn_eq, n_shared=len(shared),
                code_contra=code_contra, type_conf=type_conf, cap_conf=cap_conf, vram_conf=vram_conf, chip_conf=chip_conf,
                brand_conf=brand_conf, same_src=ra['source'] == rb['source'],
                rpm_conf=rpm_conf, mtok_contra=mtok_contra, iface_contra=iface_contra, hard_conf=mn_conf or cap_conf or rpm_conf or vram_conf or chip_conf or brand_conf)
def code_doc_freq(d):
    code_df = {}
    for cs in d.codes:
        for x in cs: code_df[x] = code_df.get(x, 0) + 1
    return code_df
