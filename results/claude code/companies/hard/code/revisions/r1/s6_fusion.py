"""Stage 6: attribute-wise fusion from final cluster members + export of all submission files."""
import os, sys, json, time, math, re, itertools, collections
import pandas as pd, numpy as np
from rapidfuzz import fuzz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from textnorm import clean_name, name_key, ascii_fold
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = f'{ROOT}/submission'
SRC_RANK = {'forbes': 0, 'dbpedia': 1, 'fullcontact': 2}
def nz(x): return x is not None and not (isinstance(x, float) and math.isnan(x))
def ocr_noisy(s):
    return bool(re.search(r'[A-Za-z][0-9]|[0-9][A-Za-z]', s)) and not re.search(r'\b\d+[A-Za-z]{1,2}\b|\b[A-Za-z]\d+\b', s)
def name_candidates(m):
    """candidate display names with provenance"""
    c = []
    for r in m.itertuples():
        if r.source == 'dbpedia' and r.title_clean:
            c.append((r.title_clean, 'dbpedia_title', r.rid, not r.native_dup))
        if r.name_clean:
            c.append((r.name_clean, f'{r.source}_name', r.rid, not r.native_dup))
        if r.source == 'forbes' and r.title:  # URL slug (lowercase) -> title-cased fallback candidate
            c.append((' '.join(w.capitalize() for w in r.title.split()), 'forbes_slug', r.rid, True))
    return c
def choose_name(m):
    cands = name_candidates(m)
    if not cands: return None, 'none'
    keys = [name_key(x[0]) for x in cands]
    best = None
    for idx, (nm, kind, rid, native) in enumerate(cands):
        support = sum(fuzz.token_sort_ratio(keys[idx], k) / 100 for j, k in enumerate(keys) if j != idx)
        pri = {'dbpedia_title': 0.3, 'forbes_name': 0.25, 'dbpedia_name': 0.1, 'fullcontact_name': 0.1, 'forbes_slug': 0.0}[kind]
        if native: pri += 0.1
        if ocr_noisy(nm): pri -= 0.5
        if kind == 'forbes_slug':
            # slug only wins when the forbes raw name disagrees with it (corrupted name)
            pri -= 0.2
        sc = support + pri
        if best is None or sc > best[0]: best = (sc, nm, kind)
    return best[1], best[2]
def vote(vals, prio):
    """vals: list of (value, source, native). majority; tie-break by source priority then native."""
    vals = [v for v in vals if nz(v[0])]
    if not vals: return None, 0, 0
    cnt = collections.Counter(v[0] for v in vals)
    def key(v):
        return (-cnt[v[0]], prio.get(v[1], 9), 0 if v[2] else 1, str(v[0]))
    b = sorted(vals, key=key)[0][0]
    return b, cnt[b], len(cnt)
TIER_W = {'grid': 1.0, 'restored': 0.9, 'text': 0.8, 'offgrid': 0.5}
SRC_W = {'forbes': 1.0, 'dbpedia': 0.7, 'fullcontact': 0.5}
def fuse_money(m, attr):
    vals = [(getattr(r, attr), getattr(r, attr + '_tier'), r.source, r.native_dup) for r in m.itertuples() if nz(getattr(r, attr))]
    if not vals: return None, 0
    groups = []
    for v in vals:
        for g in groups:
            if abs(v[0] / g[0][0] - 1) <= 0.02: g.append(v); break
        else: groups.append([v])
    def gw(g): return sum(TIER_W.get(x[1], 0.5) * SRC_W[x[2]] * (0.9 if x[3] else 1.0) for x in g)
    g = max(groups, key=lambda g: (gw(g), -min(SRC_RANK[x[2]] for x in g)))
    rep = max(g, key=lambda x: (TIER_W.get(x[1], 0.5), -SRC_RANK[x[2]]))
    return int(round(rep[0])), len(groups)
def fuse_people(m):
    lists = [(r.people, r.source, r.native_dup) for r in m.itertuples() if isinstance(r.people, list) and r.people]
    if not lists: return None
    def sim(a, b):
        A = [ascii_fold(x).lower() for x in a]; B = [ascii_fold(x).lower() for x in b]
        hits = sum(1 for x in A if any(fuzz.token_sort_ratio(x, y) >= 85 for y in B))
        return hits / max(len(A), len(B))
    best = max(lists, key=lambda l: (sum(sim(l[0], o[0]) for o in lists if o is not l), -sum(1 for p in l[0] if ocr_noisy(p)), l[1] == 'dbpedia', not l[2], len(l[0])))
    return best[0]
def main():
    R = pd.read_pickle(f'{ROOT}/work/state/records_clustered.pkl')
    rows = []; prov = []
    from types import SimpleNamespace
    class M(list):
        def itertuples(self): return iter(self)
        def __len__(self): return list.__len__(self)
    recs = [SimpleNamespace(**d) for d in R.to_dict('records')]
    groups = collections.defaultdict(list)
    for r in recs: groups[r.cluster].append(r)
    for cid in sorted(groups):
        m = M(sorted(groups[cid], key=lambda r: (r.source, r.rid)))
        name, nkind = choose_name(m)
        country, cc, cn = vote([(r.country, r.source, not r.native_dup) for r in m.itertuples()], {'forbes': 0, 'dbpedia': 1, 'fullcontact': 2})
        city, _, _ = vote([(r.city, r.source, not r.native_dup) for r in m.itertuples()], {'dbpedia': 0, 'fullcontact': 1})
        yrs = [(int(r.founded_year), r.source + ('_2d' if r.founded_2digit else ''), not r.native_dup) for r in m.itertuples() if nz(r.founded_year)]
        year, _, _ = vote(yrs, {'dbpedia': 0, 'fullcontact': 1, 'dbpedia_2d': 2, 'fullcontact_2d': 3})
        industry, _, _ = vote([(r.industry, r.source, not r.native_dup) for r in m.itertuples()], {'forbes': 0, 'dbpedia': 1})
        assets, na = fuse_money(m, 'assets'); revenue, nr = fuse_money(m, 'revenue')
        people = fuse_people(m)
        rep = sorted(m.itertuples(), key=lambda r: (r.native_dup, SRC_RANK[r.source], r.rid))[0]
        rows.append({'_id': cid, 'id': rep.rid, 'name': name, 'founded': f'{year:04d}-01-01' if year else None, 'country': country, 'city': city,
                     'industry': industry, 'assets': assets, 'revenue': revenue, 'keypeople': json.dumps(people, ensure_ascii=False) if people else None})
        prov.append({'_id': cid, 'n_members': len(m), 'sources': '|'.join(sorted({r.source for r in m})), 'name_from': nkind, 'country_variants': cn,
                     'assets_groups': na, 'revenue_groups': nr})
    F = pd.DataFrame(rows)
    os.makedirs(f'{SUB}/blocking', exist_ok=True)
    F.to_csv(f'{SUB}/fused.csv', index=False)
    pd.DataFrame(prov).to_csv(f'{ROOT}/work/state/fusion_provenance.csv', index=False)
    R[['rid', 'source', 'cluster']].rename(columns={'rid': 'record_id', 'cluster': 'cluster_id'}).to_csv(f'{SUB}/membership.csv', index=False)
    # correspondences: all cross-source pairs inside final clusters (edge score if direct edge, else min score on cluster edges)
    K = pd.read_pickle(f'{ROOT}/work/state/kept_edges.pkl')
    esc = {tuple(sorted((a, b))): s for a, b, s in zip(K.id1, K.id2, K.score)}
    src = dict(zip(R.rid, R.source))
    corr = []
    cl_of = dict(zip(R.rid, R.cluster)); cl_edges = collections.defaultdict(list)
    for k, s in esc.items(): cl_edges[cl_of[k[0]]].append(s)
    for cid, m in R.groupby('cluster'):
        if len(m) < 2: continue
        ids = sorted(m.rid)
        cl_min = min(cl_edges[cid] or [0.5])
        for a, b in itertools.combinations(ids, 2):
            if src[a] == src[b]: continue
            s = esc.get(tuple(sorted((a, b))), cl_min)
            corr.append((a, b, round(min(max(s, 0), 1.0), 4)))
    C = pd.DataFrame(corr, columns=['id1', 'id2', 'score'])
    C.to_csv(f'{SUB}/correspondences.csv', index=False)
    # blocking candidates = full candidate set + cluster-closure pairs (so that every correspondence is a candidate)
    P = pd.read_pickle(f'{ROOT}/work/state/candidates.pkl')
    cand = set(tuple(sorted(x)) for x in zip(P.id1, P.id2))
    closure = 0
    for cid, m in R.groupby('cluster'):
        if len(m) < 2: continue
        for a, b in itertools.combinations(sorted(m.rid), 2):
            if (a, b) not in cand: cand.add((a, b)); closure += 1
    pd.DataFrame(sorted(cand), columns=['id1', 'id2']).to_csv(f'{SUB}/blocking/candidates.csv', index=False)
    diag = {'ts': time.time(), 'stage': 'fusion', 'inputs': 'work/state/records_clustered.pkl', 'fused_rows': len(F), 'correspondences': len(C),
            'candidates_exported': len(cand), 'closure_pairs_added_to_candidates': closure,
            'density': {c: round(float(F[c].notna().mean()), 4) for c in F.columns}}
    print(json.dumps(diag, indent=1))
    with open(f'{ROOT}/work/diagnostics.jsonl', 'a') as fh: fh.write(json.dumps(diag) + '\n')
if __name__ == '__main__':
    main()
