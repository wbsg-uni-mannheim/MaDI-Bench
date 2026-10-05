"""Stage 6: attribute-wise fusion from each cluster's own member records + submission export."""
import pandas as pd, numpy as np, os, re, json, time, collections, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_normalize import name_key, strip_suffix, fold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREF = {  # deterministic tie-break order per attribute (first = preferred)
    'type': ['dblp', 'crossref', 'open_alex'],
    'title': ['open_alex', 'crossref', 'dblp'],
    'authors': ['open_alex', 'crossref', 'dblp'],
    'publication_year': ['crossref', 'dblp', 'open_alex'],
    'journal': ['open_alex', 'crossref', 'dblp'],
    'locator': ['crossref', 'open_alex', 'dblp'],
    'counts': ['open_alex', 'crossref', 'dblp'],
}

BIGRAM = collections.Counter()
NAMETOK = collections.Counter()

def build_corpus_stats(n):
    """Corpus-level plausibility statistics used only as tie-breakers between member values."""
    for t in n.title_k:
        w = t.split()
        BIGRAM.update(zip(w, w[1:]))
    for K in n.auth_keys:
        for k in K:
            NAMETOK.update(k.split())

def bigram_score(t):
    w = t.split()
    if len(w) < 2:
        return 0.0
    return float(np.mean([np.log1p(BIGRAM[b]) for b in zip(w, w[1:])]))

def name_score(k):
    toks = k.split()
    return float(np.mean([np.log1p(NAMETOK[t]) for t in toks])) if toks else 0.0

def rank(src, attr):
    return PREF[attr].index(src)

def vote(items, attr, key=lambda v: v):
    """items: list of (value, source). Majority on key; ties broken by source preference then value."""
    items = [(v, s) for v, s in items if v is not None and v == v and v != '']
    if not items:
        return None, 0, 0
    cnt = collections.Counter(key(v) for v, s in items)
    best_src = {}
    for v, s in items:
        k = key(v)
        best_src[k] = min(best_src.get(k, 9), rank(s, attr))
    k = sorted(cnt, key=lambda k: (-cnt[k], best_src[k], str(k)))[0]
    reps = sorted([(rank(s, attr), str(v), v) for v, s in items if key(v) == k])
    return reps[0][2], cnt[k], len(cnt)

def title_out(t, src, multi):
    t = t.strip()
    if src == 'dblp' and multi and t.endswith('.') and not t.endswith('..'):
        t = t[:-1]
    return t

def title_quality(tk_, raw):
    toks = tk_.split()
    abbrev = len(re.findall(r'\b[A-Za-z]{2,}\.(?=\s)', raw))
    return len(toks) - abbrev

def fuse_cluster(recs):
    """recs: DataFrame of member rows (normalized)."""
    out, prov = {}, {}
    srcs = recs.source.tolist()
    multi = recs.source.nunique() > 1
    # type
    v, c, k = vote(list(zip(recs.type_n, srcs)), 'type')
    out['type'] = v or ''
    prov['type'] = (c, k)
    # title: vote on normalized key; among tied keys prefer more (non-abbreviated) tokens
    items = [(tk_, raw, s) for tk_, raw, s in zip(recs.title_k, recs.title_c, srcs) if tk_]
    if items:
        cnt = collections.Counter(t for t, _, _ in items)
        q = {t: max(title_quality(t, r) for tt, r, _ in items if tt == t) for t in cnt}
        bs = {t: min(rank(s, 'title') for tt, _, s in items if tt == t) for t in cnt}
        best = sorted(cnt, key=lambda t: (-cnt[t], -q[t], -round(bigram_score(t), 6), bs[t], t))[0]
        cand = sorted([(rank(s, 'title'), r, s) for t, r, s in items if t == best])
        out['title'] = title_out(cand[0][1], cand[0][2], multi)
        prov['title'] = (cnt[best], len(cnt))
    else:
        out['title'] = ''
    # authors: pick the member list agreeing most with the others (by name keys)
    lists = [(L, K, s) for L, K, s in zip(recs.authors_c, recs.auth_keys, srcs) if L]
    if lists:
        def agree(K1, K2):
            S1, S2 = set(K1), set(K2)
            return len(S1 & S2) / max(len(S1 | S2), 1)
        scored = []
        for idx, (L, K, s) in enumerate(lists):
            sc = sum(agree(K, K2) for j, (_, K2, _) in enumerate(lists) if j != idx)
            scored.append((-round(sc, 6), rank(s, 'authors'), -len(L), idx))
        bi = sorted(scored)[0][3]
        L, K, s = lists[bi]
        names = []
        if len(lists) > 1:
            from s4_features import name_match
            for x, kx in zip(L, K):
                cands = [(x, kx, s)]
                for j, (L2, K2, s2) in enumerate(lists):
                    if j == bi:
                        continue
                    best = max(((name_match(kx, k2), y, k2) for y, k2 in zip(L2, K2)), default=(0, None, None))
                    if best[0] >= 0.8:
                        cands.append((best[1], best[2], s2))
                kc = collections.Counter(k for _, k, _ in cands)
                pick = sorted(cands, key=lambda c: (-kc[c[1]], -round(name_score(c[1]), 6), rank(c[2], 'authors'), c[0]))[0]
                names.append(strip_suffix(pick[0]) if multi else pick[0])
        else:
            names = [strip_suffix(x) if (multi or s != 'dblp') else x for x in L]
        out['authors'] = json.dumps(names, ensure_ascii=False)
        prov['authors'] = (s,)
    else:
        out['authors'] = ''
    # year
    v, c, k = vote([(y if y and 2018 <= y <= 2020 else None, s) for y, s in zip(recs.year_n, srcs)], 'publication_year')
    out['publication_year'] = '' if v is None else str(int(v))
    # journal: vote on folded key where abbreviated forms support the full form they abbreviate
    js = [(j, jk, ab, s) for j, jk, ab, s in zip(recs.journal_canon, recs.journal_ck, recs.journal_abbrev, srcs) if jk]
    if js:
        def supports(a, b):  # abbreviated key a compatible with full key b
            A, B = a.split(), b.split()
            if len(A) > len(B):
                return False
            bi = 0
            for t in A:
                while bi < len(B) and not B[bi].startswith(t):
                    bi += 1
                if bi == len(B):
                    return False
                bi += 1
            return True
        full = [x for x in js if not x[2]] or js
        score = []
        for j, jk, ab, s in full:
            sup = sum(1 for _, jk2, ab2, _ in js if jk2 == jk or (ab2 and supports(jk2, jk)))
            score.append((-sup, rank(s, 'journal'), j))
        out['journal'] = sorted(score)[0][2]
    else:
        out['journal'] = ''
    # locators
    for col in ['volume', 'issue', 'first_page', 'last_page']:
        v, c, k = vote(list(zip(recs[col + '_n'], srcs)), 'locator')
        out[col] = v or ''
    if out['first_page'] and out['last_page'] and out['first_page'].isdigit() and out['last_page'].isdigit() \
            and int(out['last_page']) < int(out['first_page']):
        # inconsistent pair: keep the first page, drop the last page unless it is the only supported value
        out['last_page'] = ''
    # counts: source preference (open_alex > crossref > dblp); these are snapshot-dependent values
    for col, tgt in [('refs_n', 'referenced_works_count'), ('cites_n', 'cited_by_count')]:
        vals = sorted([(rank(s, 'counts'), int(v)) for v, s in zip(recs[col], srcs) if v is not None and v == v])
        out[tgt] = str(vals[0][1]) if vals else ''
    return out, prov

def main():
    t0 = time.time()
    n = pd.read_pickle(f'{BASE}/work/state/s2_normalized.pkl')
    mem = pd.read_pickle(f'{BASE}/work/state/s5_membership.pkl')
    n = n.merge(mem[['record_id', 'cluster_id']], left_on='id', right_on='record_id')
    n = n.sort_values(['cluster_id', 'source', 'id'])
    build_corpus_stats(n)
    rows = []
    for cid, recs in n.groupby('cluster_id', sort=True):
        o, _ = fuse_cluster(recs)
        o['_id'] = cid
        rows.append(o)
    cols = ['_id', 'type', 'title', 'authors', 'publication_year', 'journal', 'volume', 'issue', 'first_page',
            'last_page', 'referenced_works_count', 'cited_by_count']
    fused = pd.DataFrame(rows)[cols]
    os.makedirs(f'{BASE}/submission/blocking', exist_ok=True)
    fused.to_csv(f'{BASE}/submission/fused.csv', index=False)
    mem.sort_values(['cluster_id', 'record_id'])[['record_id', 'source', 'cluster_id']].to_csv(
        f'{BASE}/submission/membership.csv', index=False)
    corr = pd.read_pickle(f'{BASE}/work/state/s5_correspondences.pkl')
    corr['score'] = corr.score.fillna(0.5).round(4)
    corr[['id1', 'id2', 'score']].sort_values(['id1', 'id2']).to_csv(f'{BASE}/submission/correspondences.csv', index=False)
    cand = pd.read_pickle(f'{BASE}/work/state/s3_candidates.pkl')
    cand = cand[cand.s1 != cand.s2][['id1', 'id2']]
    # transitive (implied) in-cluster pairs are added so that every correspondence is in the candidate set
    cand = pd.concat([cand, corr[['id1', 'id2']]]).drop_duplicates().sort_values(['id1', 'id2'])
    cand.to_csv(f'{BASE}/submission/blocking/candidates.csv', index=False)
    print('fused', fused.shape, time.time() - t0)

if __name__ == '__main__':
    main()
