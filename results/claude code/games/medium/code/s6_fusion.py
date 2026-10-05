"""Stage 6: fusion. One row per cluster, values resolved from the cluster's own members.
Per-source representative first (dbpedia contributes many cartesian-product rows per game, so each source gets one vote),
then vote across sources on a normalized key; ties -> source priority metacritic > sales > dbpedia (target schema follows
metacritic's attribute set). Invalid values (out-of-range dates, non-taxonomy ESRB) are never emitted."""
import pandas as pd, numpy as np, os, json, collections, math, re
from platforms import compact
BASE = os.path.dirname(os.path.abspath(__file__)) + '/..'
U = pd.read_pickle(f'{BASE}/work/state/s5_clusters.pkl')
surf = json.load(open(f'{BASE}/work/state/platform_surface.json'))
PRIO = {'metacritic': 0, 'sales': 1, 'dbpedia': 2}
U['_p'] = U.source.map(PRIO)
U = U.sort_values(['cluster_id', '_p', 'rid'])
# corpus token frequencies: typo'd tokens are rare, used to break ties between name spellings
tokfreq = collections.Counter(t for toks in U.ntoks for t in set(toks))
def plaus(name):
    toks = re.findall(r'[a-z0-9]+', name.lower())
    return min((tokfreq.get(t, 0) for t in toks), default=0)
def src_rep(vals, keyf, pick=None):
    """most frequent value within one source (by key), representative surface = most frequent raw of that key"""
    vals = [v for v in vals if v is not None and v == v and v != '']
    if not vals: return None
    kc = collections.Counter(keyf(v) for v in vals)
    top = max(kc.values()); keys = [k for k in kc if kc[k] == top]
    cands = [v for v in vals if keyf(v) in keys]
    if pick: return max(cands, key=pick)
    return collections.Counter(cands).most_common(1)[0][0]
def vote(per_src, keyf, pick=None):
    """per_src: list of (priority, value); majority by key, tie -> pick() then priority"""
    per_src = [(p, v) for p, v in per_src if v is not None]
    if not per_src: return None, 'missing'
    kc = collections.Counter(keyf(v) for _, v in per_src)
    top = max(kc.values()); keys = {k for k in kc if kc[k] == top}
    cands = [(p, v) for p, v in per_src if keyf(v) in keys]
    how = 'agree' if len(kc) == 1 else ('majority' if len(keys) == 1 else 'tie')
    if pick: cands.sort(key=lambda pv: (-pick(pv[1]), pv[0]))
    else: cands.sort(key=lambda pv: pv[0])
    return cands[0][1], how
# dbpedia platform spellings that are genuine source variants (known alias, >=8% of that platform's dbpedia rows),
# e.g. 'Mega Drive' vs 'Sega Genesis', 'DOS' vs 'MS-DOS'; rarer spellings are treated as noise -> dominant spelling
from platforms import ALIASES
_alias = {a for al in ALIASES.values() for a in al}
_d = U[(U.source == 'dbpedia') & U.pk.notna()]
_sh = _d.groupby('pk').platform.value_counts(normalize=True)
DBP_OK = {(pk, v) for (pk, v), x in _sh.items() if x >= 0.08 and v in _alias}
nkey_f = lambda v: ' '.join(sorted(re.findall(r'[a-z0-9]+', v.lower())))
ckey = lambda v: compact(v)
out = []; prov = []
for cid, g in U.groupby('cluster_id', sort=True):
    srcs = sorted(set(g.source), key=PRIO.get)
    by = {s: g[g.source == s] for s in srcs}
    row = {'_id': cid, 'id': cid}
    # name: metacritic/sales titles preferred (dbpedia titles carry '(video game)' disambiguators); dbpedia raw otherwise
    ms = [s for s in srcs if s != 'dbpedia']
    use = ms if ms else ['dbpedia']
    per = [(PRIO[s], src_rep(by[s].name.tolist(), nkey_f, pick=plaus)) for s in use]
    row['name'], h = vote(per, nkey_f, pick=plaus); prov.append((cid, 'name', h))
    # platform: canonical key voted across sources, emitted in the highest-priority contributing source's dominant spelling
    per = [(PRIO[s], src_rep(by[s].pk.tolist(), str)) for s in srcs]
    pk, h = vote(per, str)
    if pk is not None:
        sp = [s for s in srcs if pk in set(by[s].pk.dropna())]
        row['platform'] = surf[pk].get(sp[0]) or next(iter(surf[pk].values()))
        if sp[0] == 'dbpedia':
            own = [v for v in by['dbpedia'][by['dbpedia'].pk == pk].platform if (pk, v) in DBP_OK]
            if own: row['platform'] = collections.Counter(own).most_common(1)[0][0]
    else: row['platform'] = None
    prov.append((cid, 'platform', h))
    # release date: valid dates only (1960..2024 per schema consistency rule); vote on year
    per = []
    for s in srcs:
        b = by[s][by[s].year_valid]
        if len(b): per.append((PRIO[s], src_rep(b.date.tolist(), lambda d: d[:4], pick=lambda d: -int(d[:4]))))
    row['releaseYear'], h = vote(per, lambda d: d[:4]); prov.append((cid, 'releaseYear', h))
    for col, tgt in (('developer_n', 'developer'), ('publisher_n', 'publisher'), ('series_n', 'series')):
        per = [(PRIO[s], src_rep(by[s][col].tolist(), ckey)) for s in srcs]
        row[tgt], h = vote(per, ckey); prov.append((cid, tgt, h))
    per = [(PRIO[s], src_rep(by[s].critic.tolist(), float)) for s in srcs]
    v, h = vote(per, lambda x: round(x)); row['criticScore'] = int(v) if v is not None else None; prov.append((cid, 'criticScore', h))
    per = [(PRIO[s], src_rep(by[s].user.tolist(), float)) for s in srcs]
    row['userScore'], h = vote(per, lambda x: round(x, 1)); prov.append((cid, 'userScore', h))
    per = [(PRIO[s], src_rep(by[s].esrb.tolist(), str)) for s in srcs]
    row['ESRB'], h = vote(per, str); prov.append((cid, 'ESRB', h))
    # genres: union (set-valued attribute), ordered metacritic, sales, then dbpedia by row support; capped at schema maxItems 10
    gl = []; seen = set()
    for s in srcs:
        cnt = collections.Counter(x for l in by[s].genre_list for x in l)
        for x, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            k = compact(x)
            if k and k not in seen: seen.add(k); gl.append(x)
    row['genres'] = json.dumps(gl[:10], ensure_ascii=False) if gl else None
    out.append(row)
F = pd.DataFrame(out)
cols = ['_id', 'id', 'name', 'releaseYear', 'developer', 'genres', 'publisher', 'platform', 'criticScore', 'userScore', 'ESRB', 'series']
F = F[cols]
F['criticScore'] = F.criticScore.astype('Int64')
os.makedirs(f'{BASE}/submission', exist_ok=True)
F.to_csv(f'{BASE}/submission/fused.csv', index=False)
pd.DataFrame(prov, columns=['cluster_id', 'attribute', 'resolution']).to_csv(f'{BASE}/work/state/s6_provenance.csv', index=False)
# membership + correspondences (all cross-source record pairs inside each cluster)
U[['rid', 'source', 'cluster_id']].rename(columns={'rid': 'record_id'}).to_csv(f'{BASE}/submission/membership.csv', index=False)
P = pd.read_pickle(f'{BASE}/work/state/s4_pairs.pkl')
ps = {}
for a, b, sc, rj in P[['a', 'b', 'score', 'reject']].itertuples(index=False):
    ps[(min(a, b), max(a, b))] = (sc, rj)
corr = []
for cid, g in U.groupby('cluster_id'):
    if g.source.nunique() < 2: continue
    idx = g.index.tolist(); s = g.source.tolist(); r = g.rid.tolist()
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if s[i] == s[j]: continue
            sc, rj = ps.get((min(idx[i], idx[j]), max(idx[i], idx[j])), (None, None))
            # score: calibrated-free confidence in [0,1]; direct accepted pair -> min(1, score/100); implied by cluster -> 0.8
            conf = min(1.0, sc / 100) if sc is not None and rj is None else 0.8
            corr.append((r[i], r[j], round(max(conf, 0.5), 3)))
C = pd.DataFrame(corr, columns=['id1', 'id2', 'score'])
C.to_csv(f'{BASE}/submission/correspondences.csv', index=False)
print('fused rows', len(F), 'membership', len(U), 'correspondences', len(C))
print(F.notna().mean().round(3).to_dict())
