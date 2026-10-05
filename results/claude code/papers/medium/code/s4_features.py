"""Stage 4a: pairwise evidence features for all blocking candidates."""
import pandas as pd, numpy as np, re, sys
sys.path.insert(0, 'work')
from rapidfuzz import fuzz
u = pd.read_pickle('work/state/s2_normalized.pkl').set_index('id')
c = pd.read_pickle('work/state/s3_candidates.pkl')
from common import STOP, jtoks, venue_compat
def auth_overlap(a, b):
    if not a or not b: return np.nan
    sa, sb = set(a), set(b)
    hit = sum(1 for x in sa if x in sb or any(len(x) > 3 and fuzz.ratio(x, y) >= 80 for y in sb))
    return hit / min(len(sa), len(sb))
def tok_overlap(a, b):
    if not a or not b: return np.nan
    sa, sb = set(a), set(b); return len(sa & sb) / min(len(sa), len(sb))
from rapidfuzz.distance import Levenshtein
def title_conflicts(ta, tb):
    """(digit-token conflict on both sides, min #content tokens without a near (edit<=2) counterpart)."""
    A, B = set(ta.split()), set(tb.split())
    ca, cb = ta.replace(' ', ''), tb.replace(' ', '')   # tokens split/merged by markup ('c 1' vs 'c1') are not conflicts
    A, B = {t for t in A if t in B or t not in cb}, {t for t in B if t in A or t not in ca}
    da, db = {t for t in A - B if any(ch.isdigit() for ch in t)}, {t for t in B - A if any(ch.isdigit() for ch in t)}
    numc = float(bool(da) and bool(db))
    def lone(X, Y):
        return sum(1 for t in X - Y if t not in STOP and len(t) > 2 and not any(Levenshtein.distance(t, y) <= 2 for y in Y - X))
    return numc, min(lone(A, B), lone(B, A))
def num_eq(a, b):
    if not a or not b: return np.nan
    return float(a == b)
rows = []
import unicodedata
def fold(x):
    x = unicodedata.normalize('NFKD', x); return ''.join(ch for ch in x if not unicodedata.combining(ch)).lower()
def mainkey(t):
    h = re.split(r'\s*[:?!]\s|\s+[-–—]\s+|\s*[:]$', t, maxsplit=1)[0]
    return ' '.join(re.findall(r'[a-z0-9]+', fold(h)))
MK = {i: mainkey(t) for i, t in u.title_n.items()}
TF = u.title_key.value_counts().to_dict()   # how many records (all sources) carry exactly this title key
T = u.title_key.to_dict(); AT = u.author_tokens.to_dict(); A = u.author_keys.to_dict(); Y = u.year_n.to_dict(); J = u.journal_key.to_dict()
V = u.volume_n.to_dict(); I = u.issue_n.to_dict(); F = u.first_page_n.to_dict(); L = u.last_page_n.to_dict(); TY = u.type_n.to_dict()
for a, b in zip(c.id1, c.id2):
    ta, tb = T[a], T[b]
    r = fuzz.ratio(ta, tb) / 100.0 if ta and tb else np.nan
    s, l = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    prefix = float(len(s) >= 1 and (l.startswith(s) or l.endswith(s)) and len(s) < len(l))
    main = float(bool(ta) and bool(tb) and ta != tb and (MK[a] == tb or MK[b] == ta or (MK[a] == MK[b] and len(MK[a]) >= 3)))
    tsort = fuzz.token_sort_ratio(ta, tb) / 100.0 if ta and tb else np.nan
    numc, tokc = title_conflicts(ta, tb) if ta and tb else (np.nan, np.nan)
    ya, yb = Y[a], Y[b]
    yd = abs(int(ya) - int(yb)) if ya and yb else np.nan
    rows.append((r, prefix, main, tsort, numc, tokc, float(ta == tb and ta != ''), TF.get(s, 0), len(s), len(l), auth_overlap(A[a], A[b]), tok_overlap(AT[a], AT[b]), len(A[a]), len(A[b]), yd,
                 venue_compat(J[a], J[b]), num_eq(V[a], V[b]), num_eq(I[a], I[b]), num_eq(F[a], F[b]), num_eq(L[a], L[b]),
                 TY[a], TY[b], F[a] if F[a] == F[b] else '', V[a] if V[a] == V[b] else ''))
f = pd.DataFrame(rows, columns=['tsim','prefix','main','tsort','numc','tokc','texact','tfreq','tlen_s','tlen_l','asim','atok','na1','na2','ydiff','venue','vol_eq','iss_eq','fp_eq','lp_eq','type1','type2','fp_same','vol_same'])
f = pd.concat([c.reset_index(drop=True), f], axis=1)
f.to_pickle('work/state/s4_features.pkl')
print(f.describe().T)
