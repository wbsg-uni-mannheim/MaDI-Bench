"""Stage 2: normalization. Adds canonical comparison/fusion columns; raw columns kept untouched."""
import pandas as pd, numpy as np, re, os, json, unicodedata, collections
from rapidfuzz.distance import Levenshtein
from platforms import canon_platform, compact, toksort
BASE = os.path.dirname(os.path.abspath(__file__)) + '/..'
U = pd.read_pickle(f'{BASE}/work/state/s1_unified.pkl')
PRIO = {'metacritic': 0, 'sales': 1, 'dbpedia': 2}

def strip_acc(s): return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

# ---------- generic vocabulary denoiser (case / token-swap / split-word / 1-typo variants) ----------
def build_denoiser(values, min_ratio=3, min_len=6):
    cnt = collections.Counter(v for v in values if v)
    rep = {}
    # 1) group by sorted-token key and by compact key; representative = most frequent raw form
    for keyf in (lambda v: toksort(strip_acc(v)), lambda v: compact(strip_acc(v))):
        groups = collections.defaultdict(list)
        for v in cnt: groups[keyf(rep.get(v, v))].append(v)
        for g in groups.values():
            tot = collections.Counter()
            for v in g: tot[rep.get(v, v)] += cnt[v]
            best = max(tot, key=lambda x: (tot[x], x))
            for v in g: rep[v] = best
    rc = collections.Counter()
    for v, n in cnt.items(): rc[rep[v]] += n
    # 2) single-edit typos toward a clearly more frequent representative (same length bucket)
    reps = sorted(rc, key=lambda x: -rc[x]); comp = {r: compact(strip_acc(r)) for r in reps}
    bylen = collections.defaultdict(list)
    for r in reps: bylen[len(comp[r])].append(r)
    fix = {}
    for r in reps:
        c = comp[r]
        if len(c) < min_len: continue
        cands = [o for L in (len(c)-1, len(c), len(c)+1) for o in bylen[L]
                 if o != r and rc[o] >= min_ratio * rc[r] and Levenshtein.distance(comp[o], c, score_cutoff=1) <= 1
                 and re.sub(r'\D', '', comp[o]) == re.sub(r'\D', '', c)]   # never change digits (Studio 3 vs Studio 4)
        if cands: fix[r] = max(cands, key=lambda o: (rc[o], o))
    return {v: fix.get(rep[v], rep[v]) for v in cnt}

# ---------- platform ----------
r = U.platform.map(canon_platform)
U['pk'] = [x[0] for x in r]; U['pk_method'] = [x[1] for x in r]
unres = U.pk.isna() & U.platform.notna()
dn = build_denoiser(U.loc[unres, 'platform'].tolist())
U.loc[unres, 'pk'] = U.loc[unres, 'platform'].map(lambda v: 'x:' + compact(dn[v]))
# dominant surface form of each canonical key per source (clean forms dominate noisy ones)
surf = {}
for (src, pk), g in U[U.pk.notna()].groupby(['source', 'pk']):
    surf.setdefault(pk, {})[src] = g.platform.value_counts().index[0] if not pk.startswith('x:') else dn.get(g.platform.value_counts().index[0])
# blocking/matching platform key: unresolved metacritic/sales platform strings are treated as missing
U['pkb'] = U.pk.where(~U.pk.fillna('').str.startswith('x:') | (U.source == 'dbpedia'))
U['platform_out'] = [surf[pk][s] if isinstance(pk, str) else None for pk, s in zip(U.pk, U.source)]

# ---------- names ----------
ROMAN = {'ii':'2','iii':'3','iv':'4','v':'5','vi':'6','vii':'7','viii':'8','ix':'9','x':'10','xi':'11','xii':'12','xiii':'13',
         'xiv':'14','xv':'15','xvi':'16','xvii':'17','xviii':'18','xix':'19','xx':'20'}
PAREN = re.compile(r'\s*\(([^()]*)\)\s*$')
def clean_name(n, src):
    n = n.strip()
    if src == 'dbpedia':
        if n.startswith('('):   # shuffled 'X (YYYY video game)' -> inner tokens minus disambiguation words
            inner = n.strip('() ')
            toks = [t for t in inner.split() if not re.fullmatch(r'\d{4}|video|game|series|franchise|\(|\)', t.lower())]
            return ' '.join(toks)
        if '(' in n or ')' in n:
            m = PAREN.search(n)
            if m: n = n[:m.start()]          # trailing disambiguator: (video game), (2002 video game), (Wii), (Electronic Arts)
            else:                            # disambiguator tokens shuffled into the title
                n = ' '.join(t for t in n.split() if not re.fullmatch(r'\(?(video|game|\d{4})\)?', t, re.I) and '(' not in t and ')' not in t)
    return n
def name_tokens(n):
    n = strip_acc(n).lower().replace('&', ' and ').replace("'", '').replace('’', '')
    toks = re.findall(r'[a-z0-9]+', n)
    return [ROMAN.get(t, t) for t in toks]
U['name_clean'] = [clean_name(n, s) for n, s in zip(U.name, U.source)]
U['ntoks'] = U.name_clean.map(name_tokens)
U['nkey'] = U.ntoks.map(lambda t: ' '.join(t))
# digit runs, including those inside tokens such as '2k16', 'v4', '20-02' (sequel / edition numbers)
U['nnums'] = U.ntoks.map(lambda t: tuple(sorted(n for x in t for n in re.findall(r'\d+', x))))

# ---------- dates ----------
# letters found inside numeric fields: OCR look-alikes (O->0, l->1, S->5 ...) and keyboard-row neighbours (q->1, w->2, e->3 ... p->0)
OCR = str.maketrans({'O':'0','o':'0','l':'1','I':'1','i':'1','S':'5','s':'5','B':'8','Z':'2','z':'2','g':'9','G':'6',
                     'q':'1','w':'2','e':'3','r':'4','t':'5','y':'6','u':'7','p':'0','Q':'1','W':'2','E':'3','R':'4','T':'5','Y':'6','U':'7','P':'0'})
MONTHS = {m: i+1 for i, m in enumerate(['january','february','march','april','may','june','july','august','september','october','november','december'])}
def parse_date(v):
    if v is None: return None, None
    v = str(v).strip()
    m = re.match(r'([A-Za-z]+)\s+(\S+),\s*(\S+)$', v)
    if m and m.group(1).lower() in MONTHS:
        y = m.group(3).translate(OCR); d = m.group(2).translate(OCR)
        if re.fullmatch(r'\d{4}', y):
            dd = int(d) if d.isdigit() and 1 <= int(d) <= 31 else 1
            return int(y), f'{y}-{MONTHS[m.group(1).lower()]:02d}-{dd:02d}'
        return None, None
    w = re.sub(r'\s', '', v).translate(OCR)
    m = re.fullmatch(r'(\d{4})-?(\d{2})-?(\d{2})', w)
    if m:
        y, mo, d = m.groups()
        if not (1 <= int(mo) <= 12): mo = '01'
        if not (1 <= int(d) <= 31): d = '01'
        return int(y), f'{y}-{mo}-{d}'
    m = re.match(r'(\d{4})', w)
    if m: return int(m.group(1)), f'{m.group(1)}-01-01'
    return None, None
pd_ = U.releaseYear.map(parse_date)
U['year'] = [x[0] for x in pd_]; U['date'] = [x[1] for x in pd_]
U['year_valid'] = U.year.between(1960, 2024)

# ---------- scores ----------
DIG = OCR
def num(v):
    if v is None: return np.nan
    w = re.sub(r'\s', '', str(v)).translate(DIG).replace(',', '.')
    try: return float(w)
    except ValueError: return np.nan
U['critic'] = U.criticScore.map(num)
U['critic'] = np.floor(U.critic)   # critic scores are integers; '89.9' is a typo of '89.0'
U.loc[~U.critic.between(0, 100), 'critic'] = np.nan
U['user'] = U.userScore.map(num)
U.loc[~U.user.between(0, 10), 'user'] = np.nan

# ---------- ESRB (exhaustive taxonomy, alias K-A -> E) ----------
def esrb(v):
    if v is None: return None
    w = re.sub(r'\s', '', str(v)).upper().replace('PLUS', '+')
    w = {'EVERYONE':'E','TEEN':'T','MATURE17+':'M','MATURE':'M','EVERYONE10+':'E10+','ADULTSONLY18+':'AO','ADULTSONLY':'AO',
         'RATINGPENDING':'RP','K-A':'E','KA':'E'}.get(w, w)
    if w in ('E','E10+','T','M','AO','RP','RP-LM17'): return w
    if re.fullmatch(r'[A-Z0-9][A-Z0-9]?[0-9O]\+', w): return 'E10+'   # El0+, E1O+, S10+, D10+, E19+ : only E10+ has a '10+' suffix
    return None
U['esrb'] = U.ESRB.map(esrb)

# ---------- developer / publisher / series ----------
for col in ('developer', 'publisher', 'series'):
    vals = U[col].dropna().tolist()
    d = build_denoiser(vals)
    U[col + '_n'] = U[col].map(lambda v: d.get(v) if v else None)
    U[col + '_key'] = U[col + '_n'].map(lambda v: compact(strip_acc(v)) if v else None)
# developer tokens for similarity (multi-valued metacritic lists split on ',')
U['dev_set'] = U.developer_n.map(lambda v: frozenset(compact(strip_acc(x)) for x in v.split(',') if x.strip()) if v else frozenset())

# ---------- genres (non-exhaustive taxonomy: keep source vocabulary, denoise, dedupe) ----------
glists = U.genres.map(lambda v: [g.strip() for g in str(v).split(',') if g.strip()] if v else [])
gd = build_denoiser([g for l in glists for g in l], min_len=5)
U['genre_list'] = glists.map(lambda l: list(dict.fromkeys(gd[g] for g in l)))

U.to_pickle(f'{BASE}/work/state/s2_norm.pkl')
json.dump(surf, open(f'{BASE}/work/state/platform_surface.json', 'w'), indent=0, sort_keys=True)
print(U.groupby('source')[['pk','year','critic','user','esrb','developer_n','publisher_n','series_n']].count())
