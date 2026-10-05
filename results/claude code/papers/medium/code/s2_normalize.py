"""Stage 2: symmetric value normalization. Raw columns kept; normalized columns get suffix _n."""
import pandas as pd, re, json, unicodedata, itertools, html
from rapidfuzz.distance import Levenshtein
u = pd.read_pickle('work/state/s1_unified_raw.pkl')
schema = json.load(open('task/input/schemamatching/target_schema.json'))['properties']
TYPES = schema['type']['enum']

def fold(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def toks(s):
    return re.findall(r'[a-z0-9]+', fold(s))

# ---- type: closed vocabulary; repair typos by nearest enum value (edit distance <=2)
def norm_type(t):
    t0 = re.sub(r'[\s_]+', '', t.strip().lower())
    if not t0: return ''
    t0 = t0.replace('bookchapter', 'book-chapter').replace('postedcontent', 'posted-content')
    if t0 in TYPES: return t0
    best = min(TYPES, key=lambda e: Levenshtein.distance(t0, e))
    return best if Levenshtein.distance(t0, best) <= 2 else ''

# ---- numeric-intent repair: look-alike / keyboard-neighbour characters
LOOK = {'l':'1','I':'1','i':'1','O':'0','o':'0','B':'8','S':'5','Z':'2','z':'2'}
KEY = {'q':'12','w':'23','e':'34','r':'45','t':'56','y':'67','u':'78','i':'89','o':'90','p':'0'}
def year_cands(y):
    y = y.replace(' ', '')
    if re.fullmatch(r'\d{4}', y): return [y]
    opts = []
    for ch in y:
        if ch.isdigit(): opts.append([ch])
        else: opts.append(sorted(set(LOOK.get(ch, '') + KEY.get(ch.lower(), ''))) or ['?'])
    return [''.join(p) for p in itertools.product(*opts)]
def norm_year(y):
    if not y.strip(): return ''
    c = year_cands(y)
    inr = [x for x in c if x in ('2018','2019','2020')]
    if len(c) == 1 and re.fullmatch(r'\d{4}', c[0]):
        if c[0] in ('2018','2019','2020'): return c[0]
        # single-digit typo into range (e.g. 2920, 3020, 1019); other years kept as out-of-range
        if 2000 <= int(c[0]) <= 2030: return c[0]   # plausible real year, kept (out of target range)
        near = [t for t in ('2018','2019','2020') if sum(a != b for a, b in zip(c[0], t)) == 1]
        return near[0] if len(near) == 1 else ''
    return inr[0] if len(inr) == 1 else ''

def norm_count(v):
    v = v.strip().replace(' ', '')
    if not v or v == 'None': return ''
    ip = re.split(r'[.,]', v)[0]
    ip = ''.join(LOOK.get(ch, ch) for ch in ip)
    return str(int(ip)) if ip.isdigit() else ''

def norm_vol(v):  # volume / issue
    v = v.strip()
    if not v or v == 'None': return ''
    v = re.sub(r'(?<=\d) (?=\d)', '', v)
    if re.fullmatch(r'[0-9lIO]+', v) and re.search(r'\d', v):
        v = ''.join(LOOK.get(ch, ch) for ch in v)
    return v

def norm_page(p):
    p = p.strip()
    if not p or p == 'None': return ''
    p = re.sub(r'(?<=\d),(?=\d{3}\b)', '', p)   # 1,345 -> 1345
    p = re.sub(r'(?<=\d) +(?=[\d.])', '', p)    # '21 4' -> '214', '1 .0'
    p = re.sub(r'\.0$', '', p)
    p = re.sub(r'\.O$', '', p)
    if re.fullmatch(r'[0-9lO]+', p) and re.search(r'\d', p):
        p = ''.join(LOOK.get(ch, ch) for ch in p)
    return p

# ---- authors
def parse_authors(a):
    a = a.strip()
    if not a: return []
    if "'" in a[:3] or a.startswith('[') or '", "' in a or "', '" in a:
        s = a.strip('[]')
        parts = re.split(r"""['"]\s*,\s*['"]""", s)
        parts = [p.strip().strip('[]').strip().strip('\'"').strip() for p in parts]
    elif ';' in a:
        parts = [p.strip() for p in a.split(';')]
    else:
        s = re.sub(r',?\s+and\s+', ', ', a)
        parts = [p.strip() for p in s.split(',')]
    out = []
    for p in parts:
        p = re.sub(r'\s+\d{4}$', '', p).strip()   # dblp homonym suffix "Bing Li 0005"
        p = p.strip('[]\'" ')
        if p: out.append(p)
    return out

def name_key(n):
    t = toks(n)
    return t[-1] if t else ''

def norm_title_display(t, src):
    t = re.sub(r'\s+', ' ', html.unescape(t)).strip()
    if src == 'dblp' and t.endswith('.') and not t.endswith('..'):
        t = t[:-1]
    return t

u['type_n'] = u['type'].map(norm_type)
u['title_n'] = [norm_title_display(t, s) for t, s in zip(u.title, u.source)]
def title_key(t):
    t = re.sub(r'<mml:math.*?</mml:math>', lambda m: ' ' + re.sub(r'<[^>]+>', ' ', m.group(0)) + ' ', t, flags=re.S)
    t = re.sub(r'<[^>]+>', ' ', t)                       # html/jats tags
    t = re.sub(r'^\s*(retracted( article)?|retracted on [^:]*|retraction)\s*:\s*', '', t, flags=re.I)
    return ' '.join(toks(t))
u['title_key'] = u.title_n.map(title_key)
u['authors_l'] = u.authors.map(parse_authors)
u['authors_dirty'] = u.authors_l.map(lambda l: any(re.search(r"[\[\]'\";]", n) and not re.search(r"\w'\w", n) for n in l) or any(re.search(r"', |'\s*\w+'|\['|\]", n) for n in l))
u['author_keys'] = u.authors_l.map(lambda l: [name_key(n) for n in l if name_key(n)])
u['author_tokens'] = u.authors.map(lambda a: sorted({t for t in toks(re.sub(r'\b\d{4}\b', ' ', a)) if len(t) > 1}))
u['year_n'] = u.publication_year.map(norm_year)
u['journal_n'] = u.journal.map(lambda j: re.sub(r'\s+', ' ', html.unescape(j)).strip())
u['journal_key'] = u.journal_n.map(lambda t: ' '.join(toks(t)))
u['volume_n'] = u.volume.map(norm_vol)
u['issue_n'] = u.issue.map(norm_vol)
u['first_page_n'] = u.first_page.map(norm_page)
u['last_page_n'] = u.last_page.map(norm_page)
u['ref_n'] = u.referenced_works_count.map(norm_count)
u['cite_n'] = u.cited_by_count.map(norm_count)
u.to_pickle('work/state/s2_normalized.pkl')

# coverage / parse-failure diagnostics
rows = []
for raw, n in [('type','type_n'),('publication_year','year_n'),('volume','volume_n'),('issue','issue_n'),
               ('first_page','first_page_n'),('last_page','last_page_n'),('referenced_works_count','ref_n'),('cited_by_count','cite_n')]:
    for s, g in u.groupby('source'):
        nn = (g[raw].str.strip() != '') & (g[raw] != 'None')
        rows.append(dict(source=s, attribute=raw, non_null=int(nn.sum()), canonical=int((g[n] != '').sum()),
                         unmapped=int((nn & (g[n] == '')).sum())))
cov = pd.DataFrame(rows); cov['canonical_rate'] = (cov.canonical / cov.non_null.clip(lower=1)).round(4)
cov.to_csv('work/taxonomy_coverage.csv', index=False)
print(cov.to_string())
json.dump({'type': {'vocabulary': TYPES, 'exhaustive': True, 'policy': 'lowercase, strip spaces/underscores, nearest enum within edit distance 2, else empty'},
           'authors': {'serialization': 'JSON list in fused.csv', 'policy': 'parse list-literal or "A, B, and C"; strip DBLP 4-digit homonym suffix'},
           'publication_year': {'policy': 'look-alike/keyboard char repair constrained to 2018-2020; single-digit typo into range; else raw 4-digit kept'}},
          open('work/taxonomy_plan.json', 'w'), indent=1)
