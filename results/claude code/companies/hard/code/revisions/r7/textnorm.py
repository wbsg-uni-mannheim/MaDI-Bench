import re, unicodedata, urllib.parse
LEGAL = ['incorporated','inc','corporation','corp','company limited','co ltd','limited','ltd','llc','l l c','plc','p l c','public limited company','gmbh','gmbh co kg','ag','se','sa','s a','sarl','sas','s a s','nv','n v','bv','b v',
         'spa','s p a','oyj','oy','asa','as','a s','ab','kk','k k','pty ltd','pty','bhd','berhad','tbk','pt','co','lp','l p','llp','kgaa','ag co kgaa','srl','s r l','sab de cv','s a b de c v','de cv','cv','ltda','sl','s l','aps','a/s','ohg','kg','zao','oao','pjsc','jsc','ojsc','pao','nl','sae','saa','ptd','pte ltd','pte','holding ag']
LEGAL_RE = re.compile(r'(?:[\s,]+(?:' + '|'.join(sorted({re.escape(x).replace(r'\ ', r'[\s.]*') for x in LEGAL}, key=len, reverse=True)) + r')\.?)+\s*$', re.I)
PAREN_RE = re.compile(r'\s*\((?:company|companies|corporation|firm|business|manufacturer|automobile|automaker|car|brand|retailer|bank|airline|publisher|record label|studio|developer|video game company|video game developer|video game publisher|restaurant|chain|supermarket|store|conglomerate|holding company|[^)]*\b(?:company|firm|manufacturer|brand|retailer|group)\b[^)]*|\d{4}|[A-Z][a-z]+(?: [A-Z][a-z]+)?)\)', re.I)

def ascii_fold(s):
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()

def dbp_title(rid):
    slug = rid.rsplit('/', 1)[-1]
    if re.fullmatch(r'\d+', slug): return None
    return urllib.parse.unquote(slug).replace('_', ' ')

def forbes_slug_name(rid, url):
    for u in (rid, url):
        if isinstance(u, str):
            m = re.search(r'companies/([^/]+)/?', u)
            if m and not re.fullmatch(r'\d+', m.group(1)): return m.group(1).replace('-', ' ')
    return None

def strip_disamb(s):
    return PAREN_RE.sub('', s).strip()

TYPE_PAREN_RE = re.compile(r'\s*\((?:[^)]*\b(?:company|companies|corporation|firm|business|manufacturer|automobile|automaker|car|brand|retailer|bank|airline|publisher|label|studio|developer|restaurant|chain|supermarket|store|conglomerate|shop|magazine|retail|brewery|products|video game|comics|software|railway|film|group|holding)\b[^)]*|\d{4})\)', re.I)
def clean_name(s, aggressive=False):
    """display name: remove disambiguation parentheticals and trailing legal-form suffixes"""
    if not isinstance(s, str): return None
    s = re.sub(r'\s+', ' ', s).strip()
    s = strip_disamb(s) if aggressive else TYPE_PAREN_RE.sub('', s).strip()
    # leading shuffled disambiguator e.g. '(company) Bracco'
    s = re.sub(r'^\((?:company|automobile|manufacturer|business|firm|brand)\)\s*', '', s, flags=re.I)
    prev = None
    while prev != s:
        prev = s
        s2 = LEGAL_RE.sub('', s).strip().rstrip(',').strip()
        if s2: s = s2
    return s or None

OCR_IN_WORD = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '5': 's', '8': 'b', '4': 'a', '6': 'g', '7': 't', '9': 'g'})
STOP = {'com', 'net', 'org', 'inc', 'corporation', 'ltd', 'limited', 'llc', 'plc', 'gmbh', 'incorporated', 'the', 'and', 'of', 'group', 'holdings', 'holding', 'hldgs', 'grp', 'co', 'company', 'companies', 'corp', 'international', 'intl'}
ABBR = {'intl': 'international', "int'l": 'international', 'mgmt': 'management', 'assoc': 'associates', 'prod': 'productions', 'ent': 'entertainment',
        'rly': 'railway', 'labs': 'laboratories', 'lab': 'laboratories', 'tech': 'technologies', 'hldgs': 'holdings', 'hldg': 'holding', 'svcs': 'services',
        'natl': 'national', 'fin': 'financial', 'grp': 'group', 'mfg': 'manufacturing', 'bk': 'bank', 'ins': 'insurance', 'elec': 'electric', 'pwr': 'power',
        'dev': 'development', 'inds': 'industries', 'ind': 'industries', 'tel': 'telecom', 'comms': 'communications', 'amer': 'american', 'sys': 'systems', 'mgt': 'management', 'bros': 'brothers', 'assn': 'association', 'eng': 'engineering', 'pharm': 'pharmaceuticals',
        'chem': 'chemical', 'mkt': 'market', 'inv': 'investment', 'exch': 'exchange', 'svc': 'services', 'dist': 'distribution', 'corp': 'corporation'}
def name_tokens(s, drop_stop=False):
    if not isinstance(s, str): return []
    s = clean_name(s, aggressive=True) or ''
    s = ascii_fold(s).lower().replace('&', ' and ').replace("'s", 's')
    s = re.sub(r"[^a-z0-9 ]", ' ', s)
    toks = []
    for t in s.split():
        if re.search(r'[a-z]', t) and re.search(r'\d', t) and not re.fullmatch(r'\d+[a-z]{1,2}|[a-z]\d+|\d+[a-z]\d*', t):
            t = t.translate(OCR_IN_WORD)
        t = ABBR.get(t, t)
        toks.append(t)
    if drop_stop:
        k = [t for t in toks if t not in STOP]
        if k: toks = k
    return toks
def name_key(s, drop_stop=False):
    return ' '.join(name_tokens(s, drop_stop))
