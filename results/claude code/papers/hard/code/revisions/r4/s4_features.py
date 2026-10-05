"""Stage 4a: pairwise evidence for every blocking candidate (cross- and same-source)."""
import pandas as pd, numpy as np, os, re, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rapidfuzz.distance import Levenshtein, JaroWinkler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = {'a', 'an', 'the', 'of', 'and', 'for', 'in', 'on', 'to', 'with', 'by', 'at', 'from', 'via', 'using', 'its'}
JSTOP = {'of', 'and', 'the', 'on', 'in', 'for', 'de', 'la', 'j', 'journal'}

def tok_match(a, b):
    if a == b:
        return True
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)):
        return True   # abbreviation ('syst' ~ 'systems') / truncation
    if min(len(a), len(b)) >= 4 and Levenshtein.normalized_similarity(a, b) >= 0.75:
        return True   # typos / OCR
    return False

def soft_overlap(A, B):
    """number of tokens of A matched (1:1, greedy) in B."""
    rem = list(B)
    m = 0
    for a in A:
        if a in rem:
            rem.remove(a); m += 1; continue
        for k, b in enumerate(rem):
            if tok_match(a, b):
                del rem[k]; m += 1; break
    return m

def title_feats(t1, t2):
    A, B = t1.split(), t2.split()
    if not A or not B:
        return np.nan, np.nan, np.nan
    m = soft_overlap(A, B)
    A2, B2 = [x for x in A if x not in STOP], [x for x in B if x not in STOP]
    m2 = soft_overlap(A2, B2) if A2 and B2 else m
    # seq similarity (order-aware) on the joined keys
    seq = Levenshtein.normalized_similarity(t1, t2)
    return m / max(len(A), len(B)), (m2 / max(min(len(A2), len(B2)), 1)) if A2 and B2 else m / min(len(A), len(B)), seq

def name_match(a, b):
    """a, b: folded name keys 'first middle last'."""
    if a == b:
        return 1.0
    pa, pb = a.split(), b.split()
    if not pa or not pb:
        return 0.0
    la, lb = pa[-1], pb[-1]
    sur = max(JaroWinkler.similarity(la, lb), JaroWinkler.similarity(''.join(pa), ''.join(pb)))
    # reversed name order ('Wang Peng' vs 'Peng Wang')
    sur = max(sur, JaroWinkler.similarity(pa[0], lb) if len(pa) > 1 else 0, JaroWinkler.similarity(la, pb[0]) if len(pb) > 1 else 0)
    if sur < 0.85:
        return 0.0
    fa, fb = pa[0][0], pb[0][0]
    if len(pa) > 1 and len(pb) > 1 and fa != fb and pa[0][0] != pb[-1][0] and pb[0][0] != pa[-1][0]:
        return 0.5
    return 1.0 if sur >= 0.92 else 0.8

def author_feats(K1, K2):
    if not K1 or not K2:
        return np.nan, np.nan, np.nan
    S, L = (K1, K2) if len(K1) <= len(K2) else (K2, K1)
    rem = list(L)
    tot = 0.0
    for a in S:
        best, bi = 0.0, -1
        for k, b in enumerate(rem):
            s = name_match(a, b)
            if s > best:
                best, bi = s, k
                if s == 1.0:
                    break
        if bi >= 0 and best > 0:
            del rem[bi]; tot += best
    first = name_match(K1[0], K2[0])
    return tot / len(S), tot / len(L), first

def author_tok(K1, K2):
    """bag of name tokens (len>=2): robust to first/last names shuffled across authors."""
    if not K1 or not K2:
        return np.nan
    A = [t for k in K1 for t in k.split() if len(t) >= 2]
    B = [t for k in K2 for t in k.split() if len(t) >= 2]
    if not A or not B:
        return np.nan
    S, L = (A, B) if len(A) <= len(B) else (B, A)
    return soft_overlap(S, L) / len(S)

SEP = re.compile(r'\s*(?::|\?|!|\s-\s|\s–\s|\s—\s)\s*')

def colon_prefix(raw1, k1, raw2, k2):
    """1 if the shorter title equals the longer one's part before ':' / '?' / ' - ' (crossref truncation)."""
    if not k1 or not k2 or k1 == k2:
        return 0.0
    (rs, ks), (rl, kl) = ((raw1, k1), (raw2, k2)) if len(k1) < len(k2) else ((raw2, k2), (raw1, k1))
    parts = SEP.split(rl, maxsplit=1)
    if len(parts) == 2:
        from s2_normalize import title_key
        if title_key(parts[0]) == ks:
            return 1.0
    return 0.0

def jour_sim(j1, j2):
    if not j1 or not j2:
        return np.nan
    A = [x for x in j1.split() if x not in JSTOP]
    B = [x for x in j2.split() if x not in JSTOP]
    if not A or not B:
        return np.nan
    S, L = (A, B) if len(A) <= len(B) else (B, A)
    return soft_overlap(S, L) / len(L) if len(L) <= len(S) + 1 else soft_overlap(S, L) / len(S) * 0.9

def main():
    t0 = time.time()
    n = pd.read_pickle(f'{BASE}/work/state/s2_normalized.pkl').reset_index(drop=True)
    c = pd.read_pickle(f'{BASE}/work/state/s3_candidates.pkl')
    tk = n.title_k.values; ak = n.auth_keys.values; jk = n.journal_k.values
    yr = n.year_n.values
    tr = n.title_c.values
    rows = []
    for i, j in zip(c.i.values, c.j.values):
        t = title_feats(tk[i], tk[j])
        a = author_feats(ak[i], ak[j])
        rows.append((*t, *a, author_tok(ak[i], ak[j]), jour_sim(jk[i], jk[j]),
                     float(tk[i] == tk[j] and tk[i] != ''), colon_prefix(tr[i], tk[i], tr[j], tk[j])))
    f = pd.DataFrame(rows, columns=['t_max', 't_min', 't_seq', 'a_min', 'a_max', 'a_first', 'a_tok', 'j_sim',
                                    't_exact', 't_colon'], index=c.index)
    c = pd.concat([c, f], axis=1)
    y1, y2 = yr[c.i.values], yr[c.j.values]
    c['y_diff'] = np.abs(pd.to_numeric(pd.Series(y1), errors='coerce').values - pd.to_numeric(pd.Series(y2), errors='coerce').values)
    for col in ['volume', 'issue', 'first_page', 'last_page']:
        v1, v2 = n[col + '_n'].values[c.i.values], n[col + '_n'].values[c.j.values]
        both = (v1 != '') & (v2 != '')
        c[col[:2] + '_eq'] = np.where(both, (v1 == v2).astype(float), np.nan)
    c['len1'] = n.title_k.str.split().str.len().values[c.i.values]
    c['len2'] = n.title_k.str.split().str.len().values[c.j.values]
    c.to_pickle(f'{BASE}/work/state/s4_features.pkl')
    print(len(c), time.time() - t0)

if __name__ == '__main__':
    main()
