"""Stage 6: attribute-wise fusion over final clusters, with per-cell provenance."""
import os, re, json, unicodedata, numpy as np, pandas as pd
from collections import Counter
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
N = pd.read_pickle('work/state/s2_normalized.pkl')
M = pd.read_csv('work/state/s5_membership.csv')
assert (M.record_id.values == N.id.values).all()
N['cluster_id'] = M.cluster_id.values
schema = json.load(open('task/input/schemamatching/target_schema.json'))['properties']
ATTRS = [a for a in schema if a != 'id']
PAT = {a: re.compile(schema[a]['pattern']) for a in ATTRS if 'pattern' in schema[a]}

def fold(s):
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r'[‐-―−]', '-', s)

def vote(cands, key, prio):
    """cands: list of (source, value). Majority over key(value); ties by source priority.
    Returns (value, source, n_support, n_distinct)."""
    cands = [(s, v) for s, v in cands if v is not None and v == v]
    if not cands: return None, None, 0, 0
    cnt = Counter(key(v) for _, v in cands)
    best = max(cnt.values())
    top = [(prio.index(s), s, v) for s, v in cands if cnt[key(v)] == best]
    top.sort(key=lambda t: t[0])
    _, s, v = top[0]
    return v, s, best, len(cnt)

P_DCO = ['dblp', 'crossref', 'open_alex']
P_COD = ['crossref', 'open_alex', 'dblp']
P_OCD = ['open_alex', 'crossref', 'dblp']

def valid(a, v):
    if v is None: return False
    if a in PAT: return bool(PAT[a].match(str(v))) and len(str(v)) <= schema[a].get('maxLength', 999)
    return True

rows, prov = [], []
for cid, g in N.groupby('cluster_id', sort=True):
    recs = list(g.itertuples())
    out = {'_id': cid}; pv = {'_id': cid}
    def put(a, v, s, n, k):
        out[a] = v; pv[a] = f'{s}:{n}/{k}' if s else ''
    # type: shared dblp/crossref vocabulary; open_alex adds finer genres
    put('type', *vote([(r.source, r.type) for r in recs], lambda v: v, P_DCO))
    # title: group by comparison key; majority, then longest key (truncation/markup damage shortens), then source priority
    tc = [(r.source, r.title, r.tkey) for r in recs if r.title and r.tkey]
    if tc:
        cnt = Counter(k for _, _, k in tc)
        tc.sort(key=lambda t: (-cnt[t[2]], -len(t[2]), P_OCD.index(t[0])))
        put('title', tc[0][1], tc[0][0], cnt[tc[0][2]], len(cnt))
    else:
        put('title', None, None, 0, 0)
    akey = lambda L: frozenset(frozenset(re.findall(r'[a-z0-9]+', fold(a))) for a in L)
    v, s, n, k = vote([(r.source, r.authors if r.authors else None) for r in recs], akey, P_DCO)
    put('authors', json.dumps(v, ensure_ascii=False) if v else None, s, n, k)
    v, s, n, k = vote([(r.source, int(r.publication_year) if pd.notna(r.publication_year) else None) for r in recs], lambda v: v, P_DCO)
    put('publication_year', v, s, n, k)
    put('journal', *vote([(r.source, r.journal) for r in recs], lambda v: ' '.join(sorted(re.findall(r'[a-z0-9]+', fold(v)))), P_COD))
    for a, pr in [('volume', P_COD), ('issue', P_COD)]:
        put(a, *vote([(r.source, getattr(r, a)) for r in recs if valid(a, getattr(r, a))], lambda v: v.lower(), pr))
    # pages: take first/last as a pair from one source (crossref > open_alex > dblp, dblp pages are article-local)
    pc = [(r.source, (r.first_page, r.last_page)) for r in recs if valid('first_page', r.first_page)]
    v, s, n, k = vote(pc, lambda v: v[0].lower(), P_COD)
    if v:
        fpv, lpv = v
        if not valid('last_page', lpv) or (fpv.isdigit() and lpv.isdigit() and int(lpv) < int(fpv)): lpv = None
        put('first_page', fpv, s, n, k); put('last_page', lpv, s if lpv else None, n, k)
    else:
        put('first_page', None, None, 0, 0); put('last_page', None, None, 0, 0)
    # counts: OpenAlex-native attribute names; prefer OpenAlex, Crossref fallback
    for a in ['referenced_works_count', 'cited_by_count']:
        cands = [(r.source, int(getattr(r, a))) for r in recs if pd.notna(getattr(r, a))]
        cands.sort(key=lambda t: P_OCD.index(t[0]))
        if cands: put(a, cands[0][1], cands[0][0], 1, len({c[1] for c in cands}))
        else: put(a, None, None, 0, 0)
    rows.append(out); prov.append(pv)
F = pd.DataFrame(rows)[['_id'] + ATTRS]
for a in ['publication_year', 'referenced_works_count', 'cited_by_count']:
    F[a] = F[a].astype('Int64')
F.to_csv('submission/fused.csv', index=False)
pd.DataFrame(prov).to_csv('work/state/s6_provenance.csv', index=False)
M.to_csv('submission/membership.csv', index=False)
# correspondences: all cross-source pairs within final clusters, scored by the edge score when available
E = pd.read_pickle('work/state/s5_edges.pkl')
sc = {(a, b): s for a, b, s in zip(E.id1, E.id2, E.score)}
corr = []
for cid, g in M.groupby('cluster_id'):
    ids = g.record_id.tolist(); ss = g.source.tolist()
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            if ss[x] != ss[y]:
                a, b = ids[x], ids[y]
                corr.append((a, b, sc.get((a, b), sc.get((b, a), 0.5))))
pd.DataFrame(corr, columns=['id1', 'id2', 'score']).to_csv('submission/correspondences.csv', index=False)
print(len(F), len(corr)); print(F.notna().mean().round(3).to_dict())
