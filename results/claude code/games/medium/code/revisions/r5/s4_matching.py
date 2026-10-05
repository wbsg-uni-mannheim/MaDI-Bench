"""Stage 4: entity matching.
 1) group dbpedia rows (cartesian-product duplicates of one game/platform) into dbpedia groups
 2) score cross-source candidate pairs with interpretable evidence; hard contradictions reject
 3) aggregate to node level (metacritic record / sales record / dbpedia group)"""
import pandas as pd, numpy as np, os, re, networkx as nx
from rapidfuzz import fuzz
BASE = os.path.dirname(os.path.abspath(__file__)) + '/..'
U = pd.read_pickle(f'{BASE}/work/state/s2_norm.pkl')
C = pd.read_pickle(f'{BASE}/work/state/s3_candidates.pkl')
DI = pd.read_pickle(f'{BASE}/work/state/s3_dbp_internal.pkl')
src = U.source.values
# year given inside a dbpedia disambiguator, e.g. 'Doom (1993 video game)' -> 1993
U['dyear'] = [int(m.group(1)) if s == 'dbpedia' and (m := re.search(r'\((?:[^)]*\D)?((?:19|20)\d\d)\b', n)) else None
              for n, s in zip(U.name, U.source)]
U['dyear'] = U.dyear.astype('float')
# ---------- 1) dbpedia groups ----------
d = U[U.source == 'dbpedia']
G = nx.Graph(); G.add_nodes_from(d.index)
keyed = d[(d.nkey != '') & d.pkb.notna()]
for _, g in keyed.groupby(['pkb', 'nkey', d.dyear.fillna(-1)]):
    idx = g.index.tolist()
    G.add_edges_from(zip(idx[:-1], idx[1:]))
# typo variants of the same title: very high similarity, same numbers, same disambiguation year, long enough titles
nk, nn, dy, pkb = U.nkey.values, U.nnums.values, U.dyear.fillna(-1).values, U.pkb.values
for a, b, sr, so in DI.itertuples(index=False):
    if so >= 94 and nn[a] == nn[b] and dy[a] == dy[b] and pkb[a] == pkb[b] and min(len(nk[a]), len(nk[b])) >= 8:
        G.add_edge(a, b)
U['node'] = U.rid
for comp in nx.connected_components(G):
    comp = sorted(comp)
    U.loc[comp, 'node'] = 'dg:' + U.rid[comp[0]]
print('dbpedia rows', len(d), 'groups', U[U.source == 'dbpedia'].node.nunique())
# group-level evidence for dbpedia
gy = U[U.source == 'dbpedia'].groupby('node').year.agg(lambda s: frozenset(int(x) for x in s.dropna()))
gdev = U[U.source == 'dbpedia'].groupby('node').dev_set.agg(lambda s: frozenset().union(*s))
node_years = {n: frozenset([int(y)]) if y == y else frozenset() for n, y in zip(U.node, U.year)}
node_years.update(gy.to_dict())
node_dev = dict(zip(U.node, U.dev_set)); node_dev.update(gdev.to_dict())

# ---------- 2) pair scoring ----------
def dev_match(A, B):
    if not A or not B: return None
    for x in A:
        for y in B:
            if x == y or (min(len(x), len(y)) >= 4 and (x in y or y in x)): return True
    return False
STOP = {'the', 'a', 'an', 'of', 'and', 'in', 'on', 'for', 'to', 'vs'}
# publisher / licence prefixes that some sources prepend to a title ('Sid Meier's Civilization VI', 'Disney/Pixar Cars 3')
BRAND = {'disney', 'pixar', 'dreamworks', 'marvels', 'marvel', 'sid', 'meiers', 'tom', 'clancys', 'nickelodeon', 'nicktoons',
         'ea', 'sports', 'espn', 'wwe', 'disneys', 'pixars', 'dreamworks', 'peter', 'jacksons', 'clive', 'barkers', 'thq', 'cartoon', 'network', 'warner', 'bros', 'lucasarts', 'star', 'wars', 'mtv', 'nfl', 'fifa', 'soccer', 'the'}
# tokens that mark a different edition / version when inserted into a title
EDITION = {'hd', 'plus', 'extra', 're', 'kids', 'team', 'deluxe', 'remastered', 'remaster', 'gold', 'complete', 'ultimate',
           'special', 'collection', 'edition', 'goty', 'definitive', 'sigma', 'xtreme', 'remix', 'turbo', 'super', '3d', 'dx',
           'vr', 'new', 'jr', 'junior', 'online', 'mobile', 'pocket', 'portable', 'advance', 'party', 'tournament', 'returns', 'reloaded'}
from platforms import compact as _compact
from rapidfuzz.distance import Levenshtein as _Lev
def _close(t, u):
    if t == u or t.rstrip('s') == u.rstrip('s'): return 0
    if t.isdigit() or u.isdigit() or min(len(t), len(u)) < 4: return None
    k = 2 if min(len(t), len(u)) >= 8 else 1
    d = _Lev.distance(t, u, score_cutoff=k)
    return d if d <= k else None
def align(ta, tb):
    """one-to-one greedy alignment; returns (unmatched indices of ta, number of typo-matched tokens)"""
    used = set(); un = []; typos = 0
    for i, t in enumerate(ta):
        best = None
        for j, u in enumerate(tb):
            if j in used: continue
            d = _close(t, u)
            if d is not None and (best is None or d < best[1]): best = (j, d)
            if best and best[1] == 0: break
        if best is None:
            if t not in STOP: un.append(i)
        else:
            used.add(best[0]); typos += best[1] > 0
    return un, typos
def name_relation(a, b):
    """'exact'  : same tokens up to order / split words / plural s
       'typo'   : all tokens aligned, some with 1-2 character edits
       'prefix' : longer title has 1-2 extra known brand tokens in front
       'middle' : longer title has 1 extra non-numeric token inside (dropped word)
       'no'     : differing words on both sides, extra trailing subtitle (DLC/edition/remaster), or other"""
    ta, tb = U.ntoks[a], U.ntoks[b]
    if _compact(' '.join(ta)) == _compact(' '.join(tb)) or sorted(ta) == sorted(tb): return 'exact'
    ua, tya = align(ta, tb); ub, tyb = align(tb, ta)
    if not ua and not ub: return 'typo' if (tya or tyb) else 'exact'
    if ua and ub: return 'no'
    longer, extra = (ta, ua) if ua else (tb, ub)
    if sum(t not in STOP for t in longer) - len(extra) < 2: return 'no'     # shorter title must keep >=2 content words
    if extra == list(range(len(extra))) and len(extra) <= 2:
        return 'prefix' if all(longer[i] in BRAND for i in extra) else 'prefix_other'
    if len(extra) == 1 and 0 < extra[0] < len(longer) - 1 and longer[extra[0]] not in EDITION: return 'middle'
    return 'no'
rows = []
for a, b, set_r, sort_r, rule in C.itertuples(index=False):
    sa, sb = src[a], src[b]
    na, nb = U.node.values[a], U.node.values[b]
    ev = []; reject = None
    rel = name_relation(a, b)
    if U.nnums[a] != U.nnums[b]: reject = 'numbers'
    ya, yb = node_years[na], node_years[nb]
    dya, dyb = U.dyear[a], U.dyear[b]
    for dyv, ys in ((dya, yb), (dyb, ya)):
        if dyv == dyv and ys and min(abs(dyv - y) for y in ys) > 1: reject = reject or 'disamb_year'
    ydiff = min((abs(x - y) for x in ya for y in yb), default=None)
    if ydiff is not None:
        # metacritic vs sales: sales often carries the (earlier) Japanese release year; tolerate <=3 years if critic scores identical
        same_critic = U.critic[a] == U.critic[b]
        if 'dbpedia' not in (sa, sb) and ydiff >= 2 and not (same_critic and ydiff <= 3 and rel in ('exact', 'typo')): reject = reject or 'year'
        if ydiff == 0: ev.append('year=')
        elif ydiff >= 3 or ('dbpedia' not in (sa, sb) and ydiff >= 2): ev.append('year_far')
    ca, cb = U.critic[a], U.critic[b]
    if ca == ca and cb == cb:
        if abs(ca - cb) <= 1: ev.append('critic=')
        elif abs(ca - cb) > 2: reject = reject or 'critic'
    dm = dev_match(node_dev[na], node_dev[nb])
    if dm: ev.append('dev=')
    sort_r = fuzz.token_sort_ratio(U.nkey[a], U.nkey[b])
    if rel == 'no': reject = reject or 'name'
    if rel in ('prefix_other', 'middle') and 'year_far' in ev: reject = reject or 'partial_name_year_far'
    crit, yeq = 'critic=' in ev, 'year=' in ev
    # near-identical but distinct titles exist (Clash/Crash of the Titans, adversarial fake titles): typo matches need 2 signals
    if rel == 'typo' and not (crit or (dm and yeq) or (dm and len(U.nkey[a]) >= 15 and 'year_far' not in ev)):
        reject = reject or 'typo_uncorroborated'
    if rel == 'prefix' and not (crit or dm or yeq): reject = reject or 'prefix_uncorroborated'
    # unknown leading word (dropped 'Silent' in 'Hill: Origins' vs 'Dissidia: Final Fantasy'): needs critic or year+developer
    if rel == 'prefix_other' and not (crit or (dm and yeq)): reject = reject or 'prefix_uncorroborated'
    if rel == 'middle' and not (crit or (dm and yeq)): reject = reject or 'middle_uncorroborated'
    # ranking score for conflict resolution in clustering: name similarity + corroborating evidence
    score = sort_r + 10 * ('critic=' in ev) + 4 * ('year=' in ev) + 4 * bool(dm) - 6 * ('year_far' in ev) - 5 * (rel in ('prefix', 'prefix_other', 'middle')) - 2 * (rel == 'typo')
    rows.append((a, b, na, nb, sa, sb, rule, set_r, sort_r, rel, '|'.join(ev), reject, score))
P = pd.DataFrame(rows, columns=['a', 'b', 'na', 'nb', 'sa', 'sb', 'rule', 'set_r', 'sort_r', 'rel', 'evidence', 'reject', 'score'])
P.to_pickle(f'{BASE}/work/state/s4_pairs.pkl')
U.to_pickle(f'{BASE}/work/state/s4_nodes.pkl')
print(P.reject.value_counts(dropna=False).to_dict())
print(pd.cut(P[P.reject.isna()].score, [0, 70, 80, 85, 88, 90, 92, 95, 100, 200]).value_counts().sort_index())
