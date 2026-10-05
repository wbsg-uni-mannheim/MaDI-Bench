"""Stage 4: rule-based pairwise decisions on unit candidates (no training)."""
import pandas as pd, numpy as np, sys, re
sys.path.insert(0,'work')
from norm import *
from rapidfuzz import fuzz
U = pd.read_pickle('work/state/units.pkl').set_index('unit')
C = pd.read_pickle('work/state/candidates_units.pkl')
NUM = re.compile(r"^\d+[a-z]*$|^\d*k\d+$")
def nums(t): return tuple(sorted(re.findall(r'\d+', t.replace(' ',''))))
def year_gap(ua, ub):
    """min |year difference|; uses disambiguation years as hard hints when present."""
    ya = list(ua.years) + ([ua.dyear] if pd.notna(ua.dyear) else [])
    yb = list(ub.years) + ([ub.dyear] if pd.notna(ub.dyear) else [])
    if not ya or not yb: return None
    return min(abs(a-b) for a in ya for b in yb)
BRAND_PREFIX = {'dreamworks','disneys','disney pixar','disney','sid meiers','tim burtons','marvels','lara croft',
    'james camerons','tom clancys','nickelodeon','wwe','walt disney pictures','clive barkers','american mcgees','peter jacksons','disney pixars'}
SAME_SUFFIX = {'video game','videogame','game','remastered','remaster','hd','hd remaster','definitive edition','special edition','deluxe',
    'deluxe edition','directors cut','complete edition','game of year edition','goty edition','ultimate edition','gold edition',
    'enhanced edition','anniversary edition'}
def affix_rule(ta, tb):
    a, b = ta.split(), tb.split()
    s, l = (a, b) if len(a) < len(b) else (b, a)
    n = len(s)
    if n == 0: return None
    if l[-n:] == s and ' '.join(l[:-n]) in BRAND_PREFIX: return 'brand_prefix'
    if l[:n] == s and ' '.join(l[n:]) in SAME_SUFFIX: return 'edition_suffix'
    return None
from rapidfuzz.distance import Levenshtein
def spelling_variant(ta, tb):
    """titles equal up to token order, spacing/hyphenation, or one long-token typo."""
    a, b = ta.split(), tb.split()
    if sorted(a) == sorted(b): return 'token_order'
    na, nb = ta.replace(' ',''), tb.replace(' ','')
    if na == nb: return 'spacing'
    short_diff = len(a) == len(b) and any(x != y and min(len(x),len(y)) <= 3 for x, y in zip(a, b))
    if not short_diff and min(len(na),len(nb)) >= 10 and Levenshtein.distance(na, nb) <= 1 and na[:3] == nb[:3] and na[-3:] == nb[-3:]:
        return 'spacing'
    if len(a) == len(b) and len(a) >= 3:
        diff = [(x,y) for x,y in zip(a,b) if x != y]
        if len(diff) == 1 and min(len(diff[0][0]),len(diff[0][1])) >= 4 and Levenshtein.distance(*diff[0]) <= 2 \
           and not any(c.isdigit() for c in diff[0][0]+diff[0][1]): return 'typo'
    return None
rows=[]
for r in C.itertuples(index=False):
    a, b = U.loc[r.u1], U.loc[r.u2]
    if a.source > b.source: a, b = b, a  # order: dbpedia < metacritic < sales
    ta, tb = a.tkey, b.tkey
    exact = ta == tb
    tsr = fuzz.token_sort_ratio(ta, tb); tset = fuzz.token_set_ratio(ta, tb)
    numok = nums(ta) == nums(tb)
    yg = year_gap(a, b)
    aff = affix_rule(ta, tb) if not exact else None
    spv = spelling_variant(ta, tb) if not exact else None
    cd = abs(a.critic - b.critic) if pd.notna(a.critic) and pd.notna(b.critic) else None
    ud = abs(a.user - b.user) if pd.notna(a.user) and pd.notna(b.user) else None
    pair = a.source + '-' + b.source
    dec, rule = False, ''
    if pair == 'metacritic-sales':
        if exact and ((yg is not None and yg <= 2) or (cd is not None and cd <= 1)): dec, rule = True, 'exact+year|critic'
        elif numok and cd == 0 and ud is not None and ud <= 0.1 and yg is not None and yg <= 1 and tset >= 90: dec, rule = True, 'tset90+critic+user+year'
        elif numok and yg is not None and yg <= 1 and spv: dec, rule = True, spv+'+year'
    else:  # dbpedia vs metacritic/sales: no scores, year noisy (cross-product)
        if exact and (yg is None or yg <= 2): dec, rule = True, 'exact+year2'
        elif exact and pd.isna(a.dyear): dec, rule = True, 'exact+noyearhint'  # dbpedia year set is a noisy cross-product
        elif numok and spv and yg is not None and yg <= 1: dec, rule = True, spv+'+year'
        elif aff and yg is not None and yg <= 1: dec, rule = True, aff+'+year'
    if pair == 'metacritic-sales' and not dec and aff and yg is not None and yg <= 1 and cd is not None and cd <= 1:
        dec, rule = True, aff+'+critic+year'
    # composite score for ranking / 1:1 resolution
    score = (1.0 if exact else 0.95 if spv else 0.9 if aff else tsr/100*0.95) - 0.05*(min(yg,5) if yg is not None else 1) \
            + (0.05 if cd is not None and cd <= 1 else 0) - (0.2 if not numok else 0)
    rows.append((a.name if False else r.u1, r.u2, a.source, b.source, pair, exact, tsr, tset, numok, yg, cd, ud, round(score,4), dec, rule))
P = pd.DataFrame(rows, columns=['u1','u2','s1','s2','pair','exact','tsr','tset','numok','ygap','cdiff','udiff','score','accept','rule'])
P.to_pickle('work/state/pair_scores.pkl')
print(P.groupby('pair').accept.sum().to_dict()); print(P[P.accept].rule.value_counts().to_dict())
