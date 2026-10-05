"""Stage 6: attribute-wise fusion from final membership. Each attribute: candidates from members,
support = sum of similarity to the other members' values (medoid vote), deterministic source-priority tie-break."""
import pandas as pd, numpy as np, re, json, sys
from rapidfuzz import fuzz
sys.path.insert(0, 'work')
u = pd.read_pickle('work/state/s2_normalized.pkl')
mem = pd.read_pickle('work/state/s5_membership.pkl')
u = u.merge(mem[['record_id', 'cluster_id']], left_on='id', right_on='record_id')
from common import venue_compat
PAT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9 ._/-]*$'); PPAT = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
PRI = {  # tie-break source priority per attribute (from agreement analysis, see report)
    'type': ['crossref', 'dblp', 'open_alex'], 'title': ['crossref', 'open_alex', 'dblp'],
    'authors': ['crossref', 'open_alex', 'dblp'], 'year': ['crossref', 'dblp', 'open_alex'],
    'journal': ['crossref', 'open_alex', 'dblp'], 'num': ['crossref', 'open_alex', 'dblp'],
    'count': ['open_alex', 'crossref', 'dblp']}

def pick(cands, sim, pri):
    """cands: list of (source, value, key). Returns (value, source, support, n_distinct)."""
    cands = [c for c in cands if c[2] not in ('', None)]
    if not cands: return '', '', 0, 0
    best = None
    for s, v, k in cands:
        sup = sum(sim(k, k2) for s2, v2, k2 in cands)
        rank = (sup, -pri.index(s))
        if best is None or rank > best[0]: best = (rank, v, s)
    return best[1], best[2], round(best[0][0], 3), len({c[2] for c in cands})

eq = lambda a, b: float(a == b)
def tsim(a, b): return fuzz.ratio(a, b) / 100.0
def name_match(a, b):
    ta, tb = a, b
    if ta == tb: return True
    if not ta or not tb or ta[-1] != tb[-1]: return False   # surname must agree
    return all(any(x == y or (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y)) for y in tb) for x in ta[:-1][:1])
def auth_sim(a, b):
    if a == b: return 1.0
    hits = sum(1 for x in a if any(name_match(x, y) for y in b))
    return hits / max(len(a), len(b))
import unicodedata
def ntoks(n):
    n = unicodedata.normalize('NFKD', n); n = ''.join(ch for ch in n if not unicodedata.combining(ch)).lower()
    return tuple(re.findall(r'[a-z0-9]+', n))
def jsim(a, b):
    v = venue_compat(a, b); return 0.0 if np.isnan(v) else (1.0 if v >= 0.99 else 0.5 * v)

out, prov = [], []
for cid, g in u.groupby('cluster_id', sort=True):
    R = list(g.itertuples())
    row = {'_id': cid}; pv = {'_id': cid}
    def put(attr, res):
        row[attr] = res[0]; pv[attr] = f'{res[1]}|support={res[2]}|distinct={res[3]}'
    put('type', pick([(r.source, r.type_n, r.type_n) for r in R], eq, PRI['type']))
    put('title', pick([(r.source, r.title_n, r.title_key) for r in R], tsim, PRI['title']))
    clean = [r for r in R if r.authors_l and not r.authors_dirty] or [r for r in R if r.authors_l]
    res = pick([(r.source, json.dumps(r.authors_l, ensure_ascii=False), tuple(ntoks(n) for n in r.authors_l)) for r in clean],
               auth_sim, PRI['authors'])
    put('authors', res)
    yrs = [(r.source, r.year_n, r.year_n) for r in R if r.year_n in ('2018', '2019', '2020')] or \
          [(r.source, r.year_n, r.year_n) for r in R if r.year_n]
    put('publication_year', pick(yrs, eq, PRI['year']))
    # venue: DBLP abbreviations vote (support) but a full name is preferred as the output value
    jc = [(r.source, r.journal_n, r.journal_key) for r in R if r.journal_key]
    full = [c for c in jc if c[0] != 'dblp']
    if full:
        best = max(full, key=lambda c: (sum(jsim(c[2], k2) for _, _, k2 in jc), -PRI['journal'].index(c[0])))
        row['journal'] = best[1]; pv['journal'] = f'{best[0]}|support={sum(jsim(best[2], k2) for _, _, k2 in jc):.2f}|distinct={len({c[2] for c in jc})}'
    else:
        put('journal', pick(jc, eq, PRI['journal']))
    put('volume', pick([(r.source, r.volume_n, r.volume_n) for r in R if PAT.match(r.volume_n or '-') and len(r.volume_n) <= 32], eq, PRI['num']))
    put('issue', pick([(r.source, r.issue_n, r.issue_n) for r in R if PAT.match(r.issue_n or '-') and len(r.issue_n) <= 32], eq, PRI['num']))
    put('first_page', pick([(r.source, r.first_page_n, r.first_page_n) for r in R if PPAT.match(r.first_page_n or '-')], eq, PRI['num']))
    put('last_page', pick([(r.source, r.last_page_n, r.last_page_n) for r in R if PPAT.match(r.last_page_n or '-')], eq, PRI['num']))
    if row['first_page'].isdigit() and row['last_page'].isdigit() and int(row['last_page']) < int(row['first_page']):
        pv['last_page'] += '|dropped:last<first'; row['last_page'] = ''
    # counts: source-reported values; OpenAlex preferred when sources disagree (tie-break), see report
    put('referenced_works_count', pick([(r.source, r.ref_n, r.ref_n) for r in R], eq, PRI['count']))
    put('cited_by_count', pick([(r.source, r.cite_n, r.cite_n) for r in R], eq, PRI['count']))
    out.append(row); prov.append(pv)
cols = ['_id', 'type', 'title', 'authors', 'publication_year', 'journal', 'volume', 'issue', 'first_page', 'last_page', 'referenced_works_count', 'cited_by_count']
f = pd.DataFrame(out)[cols]
f.to_pickle('work/state/s6_fused.pkl')
pd.DataFrame(prov).to_csv('work/state/s6_provenance.csv', index=False)
print(f.shape); print((f != '').mean().round(3).to_dict())
