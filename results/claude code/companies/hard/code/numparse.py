import re, math
OCR_DIG = str.maketrans({'O':'0','o':'0','l':'1','I':'1','S':'5','B':'8'})
def parse_bare(s):
    """parse a bare locale-formatted number string -> float or None"""
    s = s.strip().replace(' ',' ')
    if not re.fullmatch(r'[0-9][0-9 .,]*', s): return None
    if ' ' in s:
        s2 = s.replace(' ', '')
        if ',' in s2 and '.' not in s2: s2 = s2.replace(',', '.')
        elif ',' in s2 and '.' in s2: s2 = s2.replace('.', '').replace(',', '.') if s2.rfind(',')>s2.rfind('.') else s2.replace(',','')
        try: return float(s2)
        except: return None
    nd, nc = s.count('.'), s.count(',')
    if nd and nc:
        if s.rfind(',') > s.rfind('.'): s = s.replace('.', '').replace(',', '.')
        else: s = s.replace(',', '')
    elif nc:
        if nc > 1 or re.fullmatch(r'\d{1,3},\d{3}', s): s = s.replace(',', '')
        else: s = s.replace(',', '.')
    elif nd > 1:
        s = s.replace('.', '')
    try: return float(s)
    except: return None

CUR = [('NT$','TWD'),('HK$','HKD'),('CA$','CAD'),('C$','CAD'),('A$','AUD'),('NZ$','NZD'),('R$','BRL'),('US$','USD'),('$','USD'),('€','EUR'),('£','GBP'),('¥','JPY'),('₹','INR')]
CODES = ['USD','EUR','GBP','CAD','AUD','NZD','INR','RMB','CNY','JPY','CHF','SGD','HKD','DKK','SEK','NOK','KRW','BRL','ZAR','TWD','MXN','RUB','TRY','PLN','ILS','CZK','MYR','THB','IDR','PHP','AED','SAR']
UNITWORDS = [('trillion',1e12),('billion',1e9),('million',1e6),('crore',1e7),('lakh',1e5),('thousand',1e3)]
def fuzzy_unit(w):
    w = w.lower().replace(' ','')
    if not w: return None
    if w in ('m','mn','mm','mil','mio'): return 1e6
    if w in ('b','bn','bil'): return 1e9
    if w in ('k',): return 1e3
    if w in ('t','tn'): return 1e12
    best=None
    for name,mult in UNITWORDS:
        # similarity by common chars / prefix
        from rapidfuzz import fuzz
        sc = fuzz.ratio(w, name)
        if w.startswith(name[:3]) or name.startswith(w[:4]): sc += 30
        if best is None or sc > best[0]: best=(sc,mult)
    return best[1] if best and best[0] >= 55 else None

def parse_money(raw):
    """returns dict(value, currency, kind) kind in {'units','bare','text','none'}; value None if unparseable.
    For bare small numbers value is the raw number (scale unresolved)."""
    if raw is None or (isinstance(raw,float) and math.isnan(raw)): return None
    s = str(raw).strip()
    if not s: return None
    b = parse_bare(s)
    if b is not None:
        return {'num': b, 'cur': None, 'mult': None, 'kind': 'bare'}
    cur = None; t = s
    for sym, code in CUR:
        if sym in t: cur = code; t = t.replace(sym, ' '); break
    for code in CODES:
        m = re.search(r'\b'+code+r'\b|^'+code+r'(?=\d)', t)
        if m: cur = cur or ('CNY' if code=='RMB' else code); t = t[:m.start()]+' '+t[m.end():]; break
    if re.match(r'^\s*R\s?\d', t) and cur is None: cur='ZAR'; t=re.sub(r'^\s*R','',t)
    m = re.search(r'(\d[\d.,]*(?:\s\d{3})*)\s*([A-Za-z][A-Za-z ]*)?', t)
    if not m or re.search(r'between|range|figure|within', s, re.I): return {'num':None,'cur':cur,'mult':None,'kind':'text'}
    num = parse_bare(m.group(1).strip())
    # OCR in the number itself e.g. '€lB million' left unparsed
    if num is None: return {'num':None,'cur':cur,'mult':None,'kind':'text'}
    unit = (m.group(2) or '').strip()
    mult = fuzzy_unit(unit) if unit else None
    if unit and mult is None:
        return {'num':None,'cur':cur,'mult':None,'kind':'text'}
    return {'num': num, 'cur': cur, 'mult': mult if mult else (1.0 if cur else None), 'kind': 'unit' if mult else 'bare'}
