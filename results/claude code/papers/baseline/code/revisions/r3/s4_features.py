"""Stage 4a: pairwise evidence for every blocking candidate."""
import os, re, numpy as np, pandas as pd
from collections import Counter
from rapidfuzz import fuzz
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
N = pd.read_pickle('work/state/s2_normalized.pkl')
C = pd.read_pickle('work/state/s3_candidates.pkl')
STOP = {'of','the','and','on','in','for','a','an','to','at','de'}
def fold_tokens(a):
    import unicodedata
    a = unicodedata.normalize('NFKD', a); a = ''.join(c for c in a if not unicodedata.combining(c)).lower()
    return [t for t in re.findall(r'[a-z0-9]+', a) if len(t) >= 2]
atoks = N.authors.map(lambda L: [set(fold_tokens(a)) for a in L]).to_numpy()
jt = N.jtok.map(lambda s: [t for t in s.split() if t not in STOP] if s else None).to_numpy()
def author_match(A, B):
    # greedy count of authors in A sharing a (>=2 char) name token with an unused author in B (order-free)
    used = set(); m = 0
    for x in A:
        for k, y in enumerate(B):
            if k not in used and x & y:
                used.add(k); m += 1; break
    return m
def jsim(a, b):
    if not a or not b: return np.nan
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    hit = sum(any(t == u or (len(t) >= 2 and u.startswith(t)) for u in l) for t in s)
    return hit / len(s)
tk = N.tkey.to_numpy(); tt = N.ttok.to_numpy(); sn = N.surnames.to_numpy()
yr = N.publication_year.astype('float').to_numpy(); src = N.source.to_numpy(); ty = N.type.to_numpy()
vol = N.volume.to_numpy(); fp = N.first_page.to_numpy(); lp = N.last_page.to_numpy(); iss = N.issue.to_numpy()
def num(x):
    return x if isinstance(x, str) and x.isdigit() else None
out = {k: [] for k in ['tratio','tjac','tcont','prefix','asim_max','asim_min','n_auth_min','first_auth','ydiff','vol_eq','fp_eq','lp_eq','iss_eq','type_conf','tlen_min','jsim']}
for i, j in zip(C.i.to_numpy(), C.j.to_numpy()):
    a, b = tk[i], tk[j]
    out['tratio'].append(fuzz.ratio(a, b) / 100 if a and b else np.nan)
    A, B = set(tt[i]), set(tt[j])
    inter = len(A & B)
    out['tjac'].append(inter / len(A | B) if A and B else np.nan)
    out['tcont'].append(inter / min(len(A), len(B)) if A and B else np.nan)
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    out['prefix'].append(bool(s) and len(s) >= 4 and l.startswith(s))
    out['tlen_min'].append(min(len(tt[i]), len(tt[j])))
    na, nb = len(atoks[i]), len(atoks[j])
    m = author_match(atoks[i], atoks[j]) if na <= nb else author_match(atoks[j], atoks[i])
    out['jsim'].append(jsim(jt[i], jt[j]))
    out['asim_max'].append(m / max(na, nb) if na and nb else np.nan)
    out['asim_min'].append(m / min(na, nb) if na and nb else np.nan)
    out['n_auth_min'].append(min(na, nb))
    out['first_auth'].append(bool(atoks[i][0] & atoks[j][0]) if na and nb else np.nan)
    out['ydiff'].append(abs(yr[i] - yr[j]))
    def eq(x, y):
        if x is None or y is None: return np.nan
        return float(str(x).lower() == str(y).lower())
    out['vol_eq'].append(eq(vol[i], vol[j])); out['iss_eq'].append(eq(iss[i], iss[j]))
    out['fp_eq'].append(eq(fp[i], fp[j])); out['lp_eq'].append(eq(lp[i], lp[j]))
    # dblp and crossref share a type vocabulary (article/inproceedings); open_alex does not
    tc = np.nan
    if {src[i], src[j]} == {'dblp', 'crossref'} and ty[i] and ty[j]:
        tc = float(ty[i] != ty[j])
    out['type_conf'].append(tc)
for k, v in out.items(): C[k] = v
C.to_pickle('work/state/s4_features.pkl')
print(C.describe().T)
