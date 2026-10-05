"""Stage 2: normalization. Symmetric canonicalization of every source; raw values kept alongside."""
import pandas as pd, re, ast, html, unicodedata, json, os
from rapidfuzz import process, fuzz

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["article", "inproceedings", "incollection", "posted-content", "review", "editorial", "letter",
         "preprint", "erratum", "book-chapter", "paratext"]
TYPE_ALIASES = {'journal-article': 'article', 'proceedings-article': 'inproceedings'}

def fold(s):
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c))

# ---------- type
KB_ROWS = ['1234567890', 'qwertyuiop', 'asdfghjkl', 'zxcvbnm']
KB_POS = {c: (r, i) for r, row in enumerate(KB_ROWS) for i, c in enumerate(row)}

def kb_close(a, b):
    """same key, or keyboard-adjacent key, or common OCR confusion."""
    if a == b:
        return True
    if (a, b) in {('0', 'o'), ('1', 'l'), ('3', 'e'), ('5', 's'), ('8', 'b'), ('1', 'i')} or \
       (b, a) in {('0', 'o'), ('1', 'l'), ('3', 'e'), ('5', 's'), ('8', 'b'), ('1', 'i')}:
        return True
    pa, pb = KB_POS.get(a), KB_POS.get(b)
    return bool(pa and pb and abs(pa[0] - pb[0]) <= 1 and abs(pa[1] - pb[1]) <= 1)

def norm_type(v):
    s = re.sub(r'\s+', '', v).lower()
    if not s:
        return ''
    if s in TYPES:
        return s
    if s in TYPE_ALIASES:
        return TYPE_ALIASES[s]
    for t in TYPES:  # truncations ('arti', 'inproceedi')
        if len(s) >= 4 and t.startswith(s):
            return t
    best = process.extractOne(s, TYPES, scorer=fuzz.ratio)
    if best and best[1] >= 60:
        return best[0]
    # keyboard-substitution noise with preserved length ('qrtifpe' -> article)
    cands = []
    for t in TYPES:
        if len(t) == len(s):
            k = sum(kb_close(x, y) for x, y in zip(s, t))
            cands.append((k / len(t), t))
    cands.sort(reverse=True)
    if cands and cands[0][0] >= 0.7 and (len(cands) == 1 or cands[1][0] < cands[0][0]):
        return cands[0][1]
    return ''

# ---------- numbers
OCR = str.maketrans({'O': '0', 'o': '0', 'l': '1', 'I': '1', 'B': '8', 'S': '5', 'G': '6', '|': '1'})
OCR_STRICT = str.maketrans({'O': '0', 'l': '1', 'I': '1', 'B': '8', 'S': '5', 'G': '6', '|': '1'})

KB_ROWS_FULL = ['1234567890-', 'qwertyuiop[', 'asdfghjkl;', 'zxcvbnm,./']

def _kb_neighbors():
    pos = {ch: (r, i) for r, row in enumerate(KB_ROWS_FULL) for i, ch in enumerate(row)}
    nb = {}
    for ch, (r, i) in pos.items():
        cand = [(r, i - 1), (r, i + 1), (r - 1, i), (r - 1, i + 1), (r + 1, i - 1), (r + 1, i)]
        nb[ch] = {KB_ROWS_FULL[a][b] for a, b in cand if 0 <= a < len(KB_ROWS_FULL) and 0 <= b < len(KB_ROWS_FULL[a])}
    return nb
KB_NB = _kb_neighbors()
OCR_D = {'O': '0', 'o': '0', 'l': '1', 'I': '1', 'B': '8', 'S': '5', 'G': '6', '|': '1'}

def decode_year(s, years=(2018, 2019, 2020, 2021)):
    """Minimum-substitution decoding of keyboard/OCR-noised 4-char years; None if not unique."""
    if len(s) != 4:
        return None
    best = []
    for y in years:
        cost = 0
        for ch, d in zip(s, str(y)):
            if ch == d:
                continue
            if OCR_D.get(ch) == d or d in KB_NB.get(ch.lower(), ()):
                cost += 1
            else:
                cost = 99; break
        best.append((cost, y))
    best.sort()
    if best[0][0] < 99 and (len(best) == 1 or best[1][0] > best[0][0]):
        return best[0][1]
    return None

def norm_year(v):
    s = re.sub(r'\s+', '', v)
    if not s:
        return None
    t = s.translate(OCR)
    if re.fullmatch(r'20(1[89]|2[01])', t):
        return int(t)
    return decode_year(s)

def norm_count(v):
    """referenced/cited counts: '49.O', '0,0', '6,813.0', '31.', '55. 0'."""
    s = re.sub(r'\s+', '', v)
    if not s or s in ('None', 'nan'):
        return None
    s = s.translate(OCR_STRICT)
    s = re.sub(r'(?<=\.)[oO]', '0', s)
    if re.fullmatch(r'\d{1,3}(,\d{3})+(\.\d*)?', s):
        s = s.replace(',', '')
    elif re.fullmatch(r'\d+,\d', s):
        s = s.replace(',', '.')
    if re.fullmatch(r'\d+(\.0*)?', s):
        return int(s.split('.')[0])
    return None

def norm_locator(v, kind):
    """volume/issue/pages. Returns (clean_value or '', flag) flag in {'', 'clean', 'noisy'}."""
    s = v.strip()
    if not s or s in ('None', 'nan'):
        return '', ''
    s = html.unescape(s)
    # digit groups with thousand separators / stray spaces: '1 864', '3,907', '6 2'
    if re.fullmatch(r'[\dOlBSGI|]+([ ,][\dOlBSGI|]+)*(\.0)?', s):
        t = re.sub(r'[ ,]', '', s).translate(OCR_STRICT)
        if re.fullmatch(r'\d+(\.0)?', t):
            if t.endswith('.0'):
                t = t[:-2]
            return t, 'clean'
    if re.fullmatch(r'\d+', s):
        return s, 'clean'
    # article locators like e1007004, D442, S12 ; issue words like 'Suppl 1', 'CSCW'
    if re.fullmatch(r'[A-Z]{1,3}\d+', s) and kind != 'volume':   # D442, W1, S12
        return s, 'clean'
    if re.fullmatch(r'e\d{4,}', s):   # e1007004 article locators
        return s, 'clean'
    if re.search(r'[A-Za-z]{3,}', s) and not re.fullmatch(r'[qwertyuiopOlBSG\d]+', s):
        return s, 'clean'
    if re.fullmatch(r'\d+[-/]\d+', s):
        return s, 'clean'
    return '', 'noisy'

# ---------- title
def title_key(t):
    s = fold(html.unescape(t)).lower()
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())

def clean_title(t):
    s = html.unescape(t).strip()
    s = re.sub(r'<[^>]+>', '', s)
    return ' '.join(s.split())

# ---------- authors
AFFIL = re.compile(r'\b(univ|university|universit|universidad|universidade|institute|institut|department|dept|'
                   r'school|college|laboratory|laboratories|lab|center|centre|corporation|inc|ltd|gmbh|faculty|'
                   r'academy|hospital|research|technology|engineering|sciences?|national|ministry)\b', re.I)
COUNTRY = {'germany', 'japan', 'china', 'usa', 'india', 'australia', 'france', 'italy', 'spain', 'korea',
           'canada', 'uk', 'united kingdom', 'united states', 'brazil', 'russia', 'netherlands', 'switzerland',
           'sweden', 'taiwan', 'singapore', 'south korea', 'p.r. china', 'pr china', 'austria', 'belgium'}

def parse_authors(v):
    s = html.unescape(v).strip()
    if not s or s in ('None', 'nan', '[]'):
        return []
    s = s.replace('<', '').replace('>', '')
    names = None
    if s.startswith('['):
        try:
            x = ast.literal_eval(s)
            if isinstance(x, list) and all(isinstance(i, str) for i in x):
                names = x
        except Exception:
            names = None
    if names is None:
        if re.search(r"['\"‘’\[\]]", s):
            t = re.sub(r"(?<=[A-Za-z])'(?=[a-z])", '\u0001', s)  # keep O'Brien-style apostrophes
            t = re.sub(r"['\"‘’“”\[\]]", ',', t).replace('\u0001', "'").replace(';', ',')
            names = t.split(',')
        else:
            t = re.sub(r',?\s+and\s+', ',', s)
            # localized final conjunctions ('A, B e C', 'A, B y C', 'A, B und C', 'A, B et C', 'A, B i C')
            if ',' in t:
                head, _, last = t.rpartition(',')
                last = re.sub(r'\s+(?:e|y|und|et|i)\s+(?=[A-Z])', ',', last)
                t = head + ',' + last
            t = t.replace(' & ', ',').replace(';', ',').replace(' | ', ',').replace(' / ', ',')
            names = t.split(',')
    out = []
    for n in names:
        n = ' '.join(n.split()).strip(" ,;")
        n = re.sub(r'^and\s+', '', n)
        if n:
            out.append(n)
    return out

def strip_suffix(n):
    """remove dblp homonym suffix ('Peng Wang 0023'), including OCR/keyboard-noised ones ('Kiu p001')."""
    m = re.search(r'\s*([0-9OoPpIlwqBGi]{3,4})$', n)
    if m and sum(ch.isdigit() for ch in m.group(1)) >= 2 and (m.start() == 0 or n[m.start()] == ' ' or m.group(1).isdigit()):
        return n[:m.start()].strip()
    return n

def is_affil(n):
    n = strip_suffix(n)
    if not n:
        return True
    if re.fullmatch(r'[\d\s.-]+', n):
        return True
    if n.strip().lower() in COUNTRY:
        return True
    if AFFIL.search(n):
        return True
    if re.search(r'\d{3,}', re.sub(r' \d{4}$', '', n)):
        return True
    return False

def name_key(n):
    """folded lowercase name with dblp homonym suffix removed."""
    s = fold(strip_suffix(n)).lower()
    s = re.sub(r'[^a-z0-9 ]+', ' ', s.replace('-', ' '))
    return ' '.join(s.split())

def surname(n):
    k = name_key(n).split()
    return k[-1] if k else ''

# ---------- journal
def journal_key(j):
    s = fold(html.unescape(j)).lower().replace('&', ' and ')
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())

def is_abbrev(j):
    return bool(re.search(r'\b[A-Za-z]{2,}\.(\s|$)', j))

def _tok_ok(a, b):
    from rapidfuzz.distance import Levenshtein as L
    return a == b or (len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a))) or \
        (min(len(a), len(b)) >= 3 and L.normalized_similarity(a, b) >= 0.6)

def journal_canon_map(jc, jk, jab):
    """Corpus-level canonical venue spelling. Within abbreviated and full forms separately:
    (1) casing/spacing variants of one key -> most frequent raw spelling;
    (2) rare keys (<=3 records) that are a character-level near-duplicate (normalized Levenshtein >= 0.8,
        every token aligned, at most one token extra) of a key >=10x more frequent -> that key."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    df = pd.DataFrame({'jc': jc, 'jk': jk, 'ab': jab})
    df = df[df.jk != '']
    out = {}
    for ab, g in df.groupby('ab'):
        rep = g.groupby('jk').jc.agg(lambda x: x.value_counts().sort_index().sort_values(ascending=False, kind='stable').index[0])
        kc = g.jk.value_counts()
        freq = kc[kc >= 10]
        fl = freq.index.tolist()
        target = {k: k for k in kc.index}
        for k, cn in kc[kc <= 3].items():
            r = process.extractOne(k, fl, scorer=Levenshtein.normalized_similarity, score_cutoff=0.8)
            if not r or freq[r[0]] < 10 * cn:
                continue
            A, B = k.split(), r[0].split()
            if abs(len(A) - len(B)) > 1:
                continue
            rem = list(B); ok = True
            for t in A:
                hit = next((i for i, b in enumerate(rem) if _tok_ok(t, b)), None)
                if hit is None:
                    ok = False; break
                del rem[hit]
            if ok:
                target[k] = r[0]
        for k in kc.index:
            out[(ab, k)] = rep[target[k]]
    return [out.get((a, k), c) if k else c for c, k, a in zip(jc, jk, jab)]

PLACEHOLDER = re.compile(r'^\s*<?\s*[A-Z]?(UNCHANGED|SAME|S?CHANGED|SAME AS INPUT|BUNCH_OF_OUTPUT)\s*>?\s*$', re.I)

def main():
    u = pd.read_pickle(f'{BASE}/work/state/s1_translated.pkl')
    # generator placeholders ('UNCHANGED', '<AUNCHANGED>', '<Same as input>', ...) carry no value: treat as missing
    u = u.copy()
    for col in ['title', 'authors', 'journal']:
        u[col] = u[col].where(~u[col].str.match(PLACEHOLDER), '')
    out = pd.DataFrame({'id': u.id, 'source': u.source})
    out['type_n'] = u['type'].map(norm_type)
    out['title_c'] = u.title.map(clean_title)
    out['title_k'] = out.title_c.map(title_key)
    auth = u.authors.map(parse_authors)
    out['authors_all'] = auth
    out['authors_c'] = auth.map(lambda L: [n for n in L if not is_affil(n)])
    out['n_affil_dropped'] = auth.map(len) - out.authors_c.map(len)
    out['auth_keys'] = out.authors_c.map(lambda L: [name_key(n) for n in L])
    out['surnames'] = out.authors_c.map(lambda L: [surname(n) for n in L if surname(n)])
    out['year_n'] = u.publication_year.map(norm_year)
    out['journal_c'] = u.journal.map(lambda j: ' '.join(html.unescape(j).split()))
    out['journal_k'] = out.journal_c.map(journal_key)
    out['journal_abbrev'] = out.journal_c.map(is_abbrev)
    out['journal_canon'] = journal_canon_map(out.journal_c.tolist(), out.journal_k.tolist(), out.journal_abbrev.tolist())
    out['journal_ck'] = out.journal_canon.map(journal_key)
    for c in ['volume', 'issue', 'first_page', 'last_page']:
        r = u[c].map(lambda v: norm_locator(v, c))
        out[c + '_n'] = r.str[0]
        out[c + '_flag'] = r.str[1]
    out['refs_n'] = u.referenced_works_count.map(norm_count)
    out['cites_n'] = u.cited_by_count.map(norm_count)
    for c in ['type', 'title', 'authors', 'publication_year', 'journal', 'volume', 'issue', 'first_page',
              'last_page', 'referenced_works_count', 'cited_by_count']:
        out['raw_' + c] = u[c]
    out.to_pickle(f'{BASE}/work/state/s2_normalized.pkl')

    # coverage / parse diagnostics
    rows = []
    for src, x in out.groupby('source'):
        for c, n, ok in [('type', 'raw_type', 'type_n'), ('publication_year', 'raw_publication_year', 'year_n'),
                         ('referenced_works_count', 'raw_referenced_works_count', 'refs_n'),
                         ('cited_by_count', 'raw_cited_by_count', 'cites_n'),
                         ('volume', 'raw_volume', 'volume_n'), ('issue', 'raw_issue', 'issue_n'),
                         ('first_page', 'raw_first_page', 'first_page_n'), ('last_page', 'raw_last_page', 'last_page_n')]:
            nn = x[n].str.strip().replace({'None': '', 'nan': ''}).ne('')
            good = x[ok].notna() & x[ok].astype(str).ne('')
            rows.append(dict(source=src, attribute=c, non_null=int(nn.sum()), canonical=int((nn & good).sum()),
                             unmapped=int((nn & ~good).sum()),
                             canonical_rate=round(float((nn & good).sum() / max(nn.sum(), 1)), 4)))
        nn = x.raw_authors.str.strip().ne('')
        rows.append(dict(source=src, attribute='authors', non_null=int(nn.sum()),
                         canonical=int((x.authors_c.map(len) > 0).sum()), unmapped=int((nn & (x.authors_c.map(len) == 0)).sum()),
                         canonical_rate=round(float((x.authors_c.map(len) > 0).sum() / max(nn.sum(), 1)), 4)))
    pd.DataFrame(rows).to_csv(f'{BASE}/work/taxonomy_coverage.csv', index=False)
    print(pd.DataFrame(rows).to_string())
    json.dump({'type': {'path': 'target_schema.properties.type.enum', 'exhaustive': True, 'values': TYPES,
                        'aliases': TYPE_ALIASES,
                        'policy': 'lowercase+remove spaces; exact enum, alias, prefix (>=4 chars) or fuzzy ratio>=60; else unmapped (empty)'},
               'authors': {'serialization': 'JSON list in fused.csv'},
               'publication_year': {'range': [2018, 2020], 'policy': 'OCR fix O/o->0,l/I->1,B->8,S->5,G->6; 2021 kept for matching but not fused outside schema range'}},
              open(f'{BASE}/work/taxonomy_plan.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
