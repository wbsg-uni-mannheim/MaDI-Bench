"""Stage 2: normalize all sources into one canonical long table work/state/records.pkl (+csv).
Raw values are kept in raw_* columns for audit; input files are never modified."""
import os, re, json, math, sys, collections
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textnorm import clean_name, name_key, dbp_title, forbes_slug_name, ascii_fold
from country import resolve as resolve_country
from numparse import parse_money
from industry import map_all as map_industry
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{ROOT}/task/input/data'

# ---------------- dates ----------------
YMAP = str.maketrans({'l':'1','i':'1','I':'1','q':'1','L':'1','o':'0','O':'0','p':'0','D':'0','w':'2','G':'6','S':'5','s':'5','B':'8','e':'3','y':'6','t':'5','u':'7','g':'9','Z':'2','z':'2'})
def parse_year(raw):
    """-> (year:int|None, two_digit:bool)"""
    if not isinstance(raw, str) or not raw.strip(): return (None, False)
    s = raw.strip().replace(' ', '')
    s2 = s.translate(YMAP)
    m = re.fullmatch(r'(\d{4})-?(\d{0,2})-?(\d{0,2})', s2) or re.fullmatch(r'(\d{4})-\d{1,2}-?\d{0,2}', s2)
    yy = None; two = False
    if m: yy = m.group(1)
    else:
        m = re.fullmatch(r'\d{1,2}[/.]\d{1,2}[/.](\d{2,4})', s2)
        if m:
            yy = m.group(1)
            if len(yy) == 2:
                two = True; y = int(yy); return ((2000 + y) if y <= 16 else (1900 + y), True)
        else:
            m = re.search(r'(1[6-9]\d\d|20[01]\d)', s2)
            if m: yy = m.group(1)
    if yy is None or len(yy) != 4: return (None, False)
    y = int(yy)
    if 1000 <= y <= 2016 and s2 == s: return (y, two)
    if 1700 <= y <= 2016: return (y, two)
    # keyboard-neighbour corruption 0<->9 : try replacing 9 by 0
    cands = []
    for i, ch in enumerate(yy):
        if ch == '9':
            c = int(yy[:i] + '0' + yy[i+1:])
            if 1700 <= c <= 2016: cands.append(c)
    if len(cands) == 1: return (cands[0], False)
    return (None, False)

# ---------------- city ----------------
REGIONS = set(x.lower().replace('_',' ') for x in '''Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada Ohio Oklahoma Oregon Pennsylvania Tennessee Texas Utah Vermont Virginia Wisconsin Wyoming
Ontario Quebec Alberta Manitoba Saskatchewan Nova_Scotia New_Brunswick British_Columbia Newfoundland Guangdong Zhejiang Jiangsu Fujian Shandong Hebei Henan Hubei Hunan Sichuan Yunnan Liaoning Jilin Heilongjiang Anhui Jiangxi Shanxi Shaanxi Gansu Guangxi Guizhou Hainan
Middlesex Surrey Kent Essex Dorset Hampshire Berkshire Yorkshire Lancashire Cheshire Devon Cornwall Sussex West_Sussex East_Sussex Hertfordshire Buckinghamshire Oxfordshire Warwickshire Lincolnshire Norfolk_County Suffolk Wiltshire Somerset Gloucestershire Greater_London Greater_Manchester West_Midlands Merseyside
New_South_Wales Queensland Tasmania Western_Australia South_Australia Bavaria Baden-Württemberg Hesse North_Rhine-Westphalia Lower_Saxony Saxony
New_York_State State_of_Illinois New_Jersey New_Mexico New_Hampshire North_Carolina South_Carolina North_Dakota South_Dakota West_Virginia Rhode_Island District_of_Columbia
England Scotland Wales Northern_Ireland United_States United_Kingdom USA US UK U.S. CA NY TX QC ON BC NSW PR Maharashtra Karnataka Tamil_Nadu QLD VIC Kerala Gujarat Punjab Haryana Uttar_Pradesh West_Bengal Telangana
Lombardy Lombardia Piedmont Veneto Tuscany Catalonia Andalusia Île-de-France Kanto Kansai Gyeonggi'''.split())
REGIONS |= {'new york state','new jersey','new mexico','new hampshire','north carolina','south carolina','north dakota','south dakota','west virginia','rhode island','district of columbia','nova scotia','new brunswick','british columbia','new south wales','western australia','south australia','west sussex','east sussex','greater london','greater manchester','west midlands','north rhine-westphalia','lower saxony','state of illinois','tamil nadu','uttar pradesh','west bengal','northern ireland','united states','united kingdom'}
CITY_ALIAS = {'nyc':'New York','sf':'San Francisco','la':'Los Angeles'}
CITY_STATES = {'hong kong','singapore','monaco','luxembourg','macau','macao','kuwait','kuwait city','panama','mexico','djibouti','andorra','san marino','vatican city','guatemala','bahrain','brunei','gibraltar'}
def _is_country(p):
    c, m = resolve_country(p)
    return c is not None and m in ('alias','key')
NOSTRIP = {'wales','england','scotland','kent','victoria','essex','surrey','devon','york'}
def split_camel(s):
    # 'OntarioToronto' -> ['Ontario','Toronto']; keep 'McDonald' style (Mc/Mac) intact
    return [p for p in re.split(r'(?<=[a-zé\.]{3})(?<!Mc)(?<!Mac)(?=[A-Z][a-z]{2})', s) if p]
def parse_city(raw, country=None):
    if not isinstance(raw, str) or not raw.strip(): return None
    s = raw.strip()
    if 'Ã' in s:
        try: s = s.encode('latin1').decode('utf8')
        except Exception: pass
    pieces = []
    for seg in re.split(r'[,;/]|\s-\s', s):
        seg = seg.strip()
        if not seg: continue
        # camel-joined pieces
        for p in split_camel(seg) if ' ' not in seg or re.search(r'[a-z][A-Z]', seg) else [seg]:
            p = p.strip(' .')
            if p: pieces.append(p)
    pieces = [re.sub(r'\((?:state|city|province|country)\)', '', p, flags=re.I).strip() for p in pieces]
    pieces2 = []
    for p in pieces:
        toks = p.split()
        # shuffled 'Texas Dallas' / 'Santa Clara CA': drop leading/trailing region token(s)
        while len(toks) > 1 and toks[0].lower() in REGIONS and toks[0].lower() not in NOSTRIP: toks = toks[1:]
        while len(toks) > 1 and toks[-1].lower() in REGIONS and toks[-1].lower() not in NOSTRIP: toks = toks[:-1]
        if toks: pieces2.append(' '.join(toks))
    pieces = pieces2
    if not pieces: return None
    pieces = [p for p in pieces if not re.search(r'\b(avenue|street|road|building|buildi|district|plaza|tower|floor|county|prefecture|oblast|province|governorate|region)\b', p, re.I)] or pieces
    cands = [p for p in pieces if p.lower() not in REGIONS and (p.lower() in CITY_STATES or not _is_country(p)) and len(p) > 1]
    if not cands: cands = [p for p in pieces if len(p) > 1 and (p.lower() in CITY_STATES or not _is_country(p))]
    if not cands: return None
    c = cands[0]
    c = re.sub(r'\s+', ' ', c)
    if c.lower() in CITY_ALIAS: c = CITY_ALIAS[c.lower()]
    if c.isupper() and len(c) > 3: c = c.title()
    return c

_KB = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
def _kb_adjacent(a, b):
    for r, row in enumerate(_KB):
        if a in row:
            i = row.index(a)
            for rr in (r - 1, r, r + 1):
                if 0 <= rr < 3:
                    for j in (i - 1, i, i + 1):
                        if 0 <= j < len(_KB[rr]) and _KB[rr][j] == b: return True
    return False
_OCR_PAIRS = {frozenset(p) for p in ['oa', 'ce', 'hb', 'hn', 'il', 'uv', 'gq', 'rn', 'ec', 'ao', 'ij', 'mn']}
def _noise_variant(x, y):
    """y is a plausible OCR/keyboard corruption of x: one substitution (not first char) by an OCR-confusable or
    keyboard-adjacent character, or one adjacent transposition."""
    if len(x) != len(y) or len(x) < 5 or x[0] != y[0]: return False
    diff = [i for i in range(len(x)) if x[i] != y[i]]
    if len(diff) == 1:
        a, b = x[diff[0]], y[diff[0]]
        return frozenset((a, b)) in _OCR_PAIRS or _kb_adjacent(a, b) or (a.isdigit() or b.isdigit())
    if len(diff) == 2 and diff[1] == diff[0] + 1 and x[diff[0]] == y[diff[1]] and x[diff[1]] == y[diff[0]]: return True
    return False
def canon_cities(col):
    """Corpus-frequency spelling repair of rare city strings (freq<=1) towards a frequent city (freq>=5):
    same string without spaces ('Minneap olis'), a unique OCR/keyboard single-character variant ('Modrid'->'Madrid'),
    or a unique truncation (>=4 chars, <=4 missing: 'Lake Fores'->'Lake Forest')."""
    k = lambda c: ascii_fold(c).lower().replace(' ', '').replace('-', '')
    vals = [c for c in col if isinstance(c, str)]
    freq = collections.Counter(k(c) for c in vals)
    rep = {}
    for c in vals: rep.setdefault(k(c), collections.Counter())[c] += 1
    frequent = [x for x, n in freq.items() if n >= 5]
    # a frequent key that is itself a truncation of a more frequent key is not a valid target
    frequent = [x for x in frequent if not any(y != x and y.startswith(x) and freq[y] > freq[x] for y in frequent)]
    best_form = {x: sorted(rep[x].items(), key=lambda t: (-t[1], ' ' in t[0].strip() and len(t[0].split()) > len(t[0].replace(' ', '')) // 6, t[0]))[0][0] for x in frequent}
    fix = {}
    for c in set(vals):
        kc = k(c)
        if kc in best_form:
            if c != best_form[kc]: fix[c] = best_form[kc]   # spacing/case variants: 'Minneap olis', 'Z ug', 'SHENZHEN'
            continue
        if freq[kc] > 1 or len(kc) < 4: continue
        cands = [x for x in frequent if _noise_variant(x, kc)]
        if len(cands) != 1:
            cands = [x for x in frequent if x.startswith(kc) and 0 < len(x) - len(kc) <= 4]
        if len(cands) == 1: fix[c] = best_form[cands[0]]
    return col.map(lambda c: fix.get(c, c) if isinstance(c, str) else c)

# ---------------- key people ----------------
ROLE = r'(?:co-?founders?|founders?|founded by|chief\s+\w+(?:\s+\w+)?\s+officer|chief|officer|ceo|cto|cfo|coo|cmo|chairman|chairwoman|chair|chairperson|president|vice[- ]president|vp|managing director|director|executive|head of [\w ]+|creative director|operations|manager|owner|partner|principal|md|secretary|treasurer|board|lead|leader|serves as|is|while|dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|sir)'
GENERIC = re.compile(r'\b(team|board|leadership|executive|directors|management|committee|politics|heads|headed|led|fictional|unknown|various|n/a|none)\b', re.I)
from rapidfuzz import fuzz as _fz
ROLEWORDS = ['chief','officer','executive','operating','financial','founder','cofounder','director','chair','chairman','chairwoman','president','managing','operations','technology','technical','creative','marketing','product','head','lead','manager','partner','serves','ceo','cto','cfo','coo']
PARTICLES = {'de','van','von','da','del','der','den','bin','al','el','le','la','du','di','dos','das','ter','ten','y','of','the','&'}
_PFIX = str.maketrans({'1':'l','0':'o','5':'s','3':'e','4':'a','6':'G','8':'B','7':'T'})
def _roleish(t):
    k = ascii_fold(t).lower().translate(str.maketrans({'1':'l','0':'o','3':'e','5':'s','4':'a','6':'g','8':'b'})).strip('.,:;')
    if len(k) < 3: return k in ('ceo',)
    return any(_fz.ratio(k, w) >= 80 for w in ROLEWORDS)
GENERIC_WORDS = ['board','directors','team','leadership','management','executive','committee','founders','shareholders','members','staff','unknown']
def _genericish(p):
    toks = [ascii_fold(t).lower().translate(str.maketrans({'1':'l','0':'o','3':'e','5':'s','4':'a','6':'g','8':'b'})).strip('.,:;') for t in p.split()]
    return any(_fz.ratio(t, w) >= 80 for t in toks for w in GENERIC_WORDS if len(t) >= 4)
def parse_people(raw):
    if not isinstance(raw, str) or not raw.strip(): return []
    s = raw.strip()
    s = re.sub(r"^\[|\]$", '', s)
    s = re.sub(r"\([^)]*\)", lambda m: ' ' if any(_roleish(t) for t in re.findall(r'[\w.]+', m.group(0))) or len(m.group(0)) <= 5 else m.group(0), s)  # '(CEO)'
    parts = re.split(r"[;\n]|',\s*'|\"\s*,\s*\"|\s+and\s+|\s*&\s*|,\s*(?=[A-Z][a-z]+\s+[A-Z])|,", s)
    out = []
    for p in parts:
        p = p.strip(" '\"[]")
        if not p: continue
        if ':' in p: p = p.split(':', 1)[1].strip()
        p = re.sub(r'\b' + ROLE + r'\b', ' ', p, flags=re.I)
        p = re.sub(r'\s+', ' ', p).strip(" ,.-'\"")
        if not p or GENERIC.search(p) or _genericish(p): continue
        toks = [t.translate(_PFIX) if re.search(r'[A-Za-z]', t) and re.search(r'\d', t) else t for t in p.split()]
        toks = [t for t in toks if not _roleish(t)]
        toks = [t for t in toks if t[:1].isupper() or t.lower() in PARTICLES or re.match(r'[A-ZÀ-Ý]', t)]
        while toks and toks[-1].lower() in PARTICLES: toks = toks[:-1]
        while toks and toks[0].lower() in PARTICLES - {'de', 'van', 'von', 'al', 'el', 'le', 'la', 'di', 'da', 'du'}: toks = toks[1:]
        toks = [t.translate(_PFIX) if re.search(r'[A-Za-z]', t) and re.search(r'\d', t) else t for t in toks]
        p = ' '.join(toks).strip(" ,.-'\"")
        if not (1 <= len(toks) <= 5): continue
        if not re.match(r'[A-ZÀ-Ý]', p): continue
        out.append(p)
    # dedupe preserve order
    seen = set(); res = []
    for p in out:
        k = ascii_fold(p).lower()
        if k not in seen: seen.add(k); res.append(p)
    return res

# ---------------- money ----------------
MAXV = {'assets': 4e12, 'revenue': 1e12}
PLAUS = {'assets': 1.5e12, 'revenue': 3e11}
FACTORS = (0.7842, 0.9215)
def money_value(raw, attr, src):
    """-> dict(v, tier, note, cands). Bare numbers: x>=1e7 units; x<1000 billions; else millions.
    Values off the 0.1bn grid whose division by a perturbation factor (0.7842 / 0.9215) lands on the grid within
    rendering precision are restored. 'cands' keeps weighted alternatives (all fitting factors, raw value, alternative
    scale for small dbpedia numbers) so that fusion can use agreement with other cluster members."""
    p = parse_money(raw)
    if p is None: return {'v': None, 'tier': None, 'note': 'missing', 'cands': []}
    if p['num'] is None: return {'v': None, 'tier': None, 'note': 'unparseable', 'cands': []}
    x = p['num']; cands = []
    if p['kind'] == 'unit' or p['mult']:
        v = x * (p['mult'] or 1.0)
        note = f"unit:{p['cur'] or ''}"
        tier = 'text'; cands = [(v, 0.8)]
    else:
        s = str(raw).strip()
        dec = re.search(r'[.,](\d{1,2})$', s)
        ndec = len(dec.group(1)) if dec else 0
        if x >= 1e7: scale = 1.0
        elif x < 1000: scale = 1e9
        else: scale = 1e6
        if x * scale > MAXV[attr] and scale == 1e9: scale = 1e6
        # dbpedia hosts many small firms: a bare <1000 value read as billions that would exceed a top-10-global magnitude is read as millions
        if src == 'dbpedia' and scale == 1e9 and x * scale > PLAUS[attr]: scale = 1e6
        v = x * scale
        prec = 0.5 * (10 ** -ndec) * scale  # rendering half-precision in USD
        note = f'bare:x{scale:.0e}'
        G = 1e8 if src == 'forbes' else 1e6   # observed granularity of unperturbed values (Forbes: 0.1bn; DBpedia: 1m)
        tier = 'grid' if abs(v / G - round(v / G)) * G <= max(prec, 1) else 'offgrid'
        if tier == 'grid':
            cands = [(v, 1.0)]
        else:
            fits = []
            for f in FACTORS:
                c = v / f; dev = abs(c / G - round(c / G)) * G
                tol = max(prec / f, 1) + 1e-6 * c
                if dev <= tol and round(c / G) > 0 and tol < G / 4: fits.append((dev / tol, f, round(c / G) * G))
            fits.sort()
            trusted = prec <= 1e6 or src == 'forbes'
            for k, (_, f, c) in enumerate(fits):
                cands.append((c, (0.95 if k == 0 else 0.85) if trusted else 0.5))
            cands.append((v, 0.4 if (fits and trusted) else 0.6))
            if fits and trusted:
                v = fits[0][2]; tier = 'restored'; note += f':restored/{fits[0][1]}' + (f'|alt/{fits[1][1]}' if len(fits) > 1 else '')
            elif fits:
                note += f':maybe/{fits[0][1]}={fits[0][2]:.0f}'
        if src == 'dbpedia' and scale == 1e9:
            cands.append((x * 1e6, 0.3))
    cands = [(c, w) for c, w in cands if 0 <= c <= MAXV[attr]]
    if v is not None and (v < 0 or v > MAXV[attr]):
        return {'v': None, 'tier': None, 'note': note + ':out_of_range', 'cands': cands}
    return {'v': v, 'tier': tier, 'note': note, 'cands': cands}

def main():
    dbp = pd.read_csv(f'{D}/dbpedia.csv', dtype=str, keep_default_na=False, na_values=[''])
    frb = pd.read_csv(f'{D}/forbes.csv', dtype=str, keep_default_na=False, na_values=[''])
    fco = pd.read_csv(f'{D}/fullcontact.csv', dtype=str, keep_default_na=False, na_values=[''])
    rows = []
    for _, r in dbp.iterrows():
        title = dbp_title(r['id'])
        rows.append(dict(source='dbpedia', rid=r['id'], raw_name=r['nm'], title=title, raw_country=r['cn'], raw_city=r['hq'],
                         raw_founded=r['ey'], raw_industry=r['sg'], raw_people=r['kpn'], raw_assets=r['ta'], raw_revenue=r['ai'],
                         native_dup=title is None, link=None))
    for _, r in frb.iterrows():
        slug = forbes_slug_name(r['id'], r['url'])
        num = bool(re.search(r'companies/\d+/?$', r['id']))
        link = None
        if isinstance(r['url'], str) and r['url'].rstrip('/') != r['id'].rstrip('/'):
            link = r['url'] if r['url'].endswith('/') else r['url'] + '/'
        rows.append(dict(source='forbes', rid=r['id'], raw_name=r['co_nm'], title=slug, raw_country=r['region'], raw_city=None,
                         raw_founded=None, raw_industry=r['bus_seg'], raw_people=None, raw_assets=r['ast_val'], raw_revenue=r['sls_fig'],
                         native_dup=num, link=link))
    for _, r in fco.iterrows():
        rows.append(dict(source='fullcontact', rid=r['id'], raw_name=r['Attribute_2'], title=None, raw_country=r['Attribute_3'], raw_city=r['Attribute_4'],
                         raw_founded=r['Attribute_6'], raw_industry=None, raw_people=r['Attribute_5'], raw_assets=None, raw_revenue=None,
                         native_dup=False, link=None))
    R = pd.DataFrame(rows)
    R = R.where(pd.notna(R), None)
    # names
    R['name_clean'] = R.raw_name.map(clean_name)
    R['title_clean'] = R.title.map(lambda t: clean_name(t) if t else None)
    R['nkey'] = R.raw_name.map(lambda s: name_key(s))
    R['nkey_core'] = R.raw_name.map(lambda s: name_key(s, True))
    R['tkey'] = R.title.map(lambda s: name_key(s) if s else '')
    R['tkey_core'] = R.title.map(lambda s: name_key(s, True) if s else '')
    # country
    cc = R.raw_country.map(resolve_country)
    R['country'] = cc.map(lambda x: x[0]); R['country_method'] = cc.map(lambda x: x[1])
    R['city'] = R.raw_city.map(parse_city)
    R['city'] = canon_cities(R['city'])
    fy = R.raw_founded.map(parse_year)
    R['founded_year'] = fy.map(lambda x: x[0]); R['founded_2digit'] = fy.map(lambda x: x[1])
    imap = map_industry(R.raw_industry.dropna().tolist())
    R['industry'] = R.raw_industry.map(lambda x: imap[x][0] if isinstance(x, str) and x in imap else None)
    R['industry_method'] = R.raw_industry.map(lambda x: imap[x][1] if isinstance(x, str) and x in imap else None)
    R['people'] = R.raw_people.map(parse_people)
    for a in ['assets', 'revenue']:
        mv = [money_value(x, a, s) for x, s in zip(R['raw_' + a], R.source)]
        R[a] = [m['v'] for m in mv]; R[a + '_cands'] = [m['cands'] for m in mv]; R[a + '_tier'] = [m['tier'] for m in mv]; R[a + '_note'] = [m['note'] for m in mv]
    R.to_pickle(f'{ROOT}/work/state/records.pkl')
    R.to_csv(f'{ROOT}/work/state/records.csv', index=False)
    # taxonomy coverage + parse diagnostics
    cov = []
    for (src, attr, rawc) in [(s, a, rc) for s in ['dbpedia', 'forbes', 'fullcontact'] for a, rc in [('country', 'raw_country'), ('industry', 'raw_industry'), ('city', 'raw_city'), ('founded_year', 'raw_founded'), ('assets', 'raw_assets'), ('revenue', 'raw_revenue'), ('people', 'raw_people')]]:
        sub = R[R.source == src]
        nn = sub[rawc].notna().sum()
        if nn == 0: continue
        ok = sub[attr].map(lambda v: bool(v) if isinstance(v, list) else v is not None and not (isinstance(v, float) and math.isnan(v))).sum()
        cov.append(dict(source=src, attribute=attr, non_null=int(nn), canonical=int(ok), unmapped=int(nn - ok), canonical_rate=round(ok / nn, 4)))
    pd.DataFrame(cov).to_csv(f'{ROOT}/work/taxonomy_coverage.csv', index=False)
    print(pd.DataFrame(cov).to_string())
    json.dump({'country': {'path': 'task/input/schemamatching/CLDR_Country_Taxonomy.csv', 'canonical_column': 'Country Name', 'aliases': ['Country Short Name', 'Country Variant Name'], 'exhaustive': True,
                           'policy': 'alias/key/confusable/anagram/prefix/fuzzy match (work/country.py); unresolvable -> null'},
               'industry': {'path': 'task/input/schemamatching/GICS_Industry_Taxonomy.csv', 'canonical_column': 'Industry Name', 'exhaustive': True,
                            'policy': 'manual override dictionary (work/industry.py) -> fuzzy override -> embedding NN over sub-industry names; always a GICS Industry Name'},
               'keypeople': {'serialization': 'JSON list'}}, open(f'{ROOT}/work/taxonomy_plan.json', 'w'), indent=1)
if __name__ == '__main__':
    main()
