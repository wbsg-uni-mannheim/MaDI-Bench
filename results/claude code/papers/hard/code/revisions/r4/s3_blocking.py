"""Stage 3: blocking. Union of (a) title word TF-IDF kNN, (b) title char-4gram TF-IDF kNN,
(c) author-surname TF-IDF kNN.  Pairs within the same source are kept too (used only for
within-source duplicate detection; correspondences are cross-source)."""
import pandas as pd, numpy as np, os, json, time
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K_TITLE, K_CHAR, K_AUTH = 12, 8, 10
MIN_SIM = {'title': 0.25, 'char': 0.30, 'auth': 0.30}

def knn(X, k, min_sim, chunk=2000):
    X = X.tocsr().astype(np.float32)
    XT = X.T.tocsc()
    n = X.shape[0]
    I, J, S = [], [], []
    for s in range(0, n, chunk):
        M = (X[s:s + chunk] @ XT).tocsr()
        for r in range(M.shape[0]):
            a, b = M.indptr[r], M.indptr[r + 1]
            cols, vals = M.indices[a:b], M.data[a:b]
            keep = (cols != s + r) & (vals >= min_sim)
            cols, vals = cols[keep], vals[keep]
            if len(vals) > k:
                top = np.argpartition(-vals, k)[:k]
                cols, vals = cols[top], vals[top]
            I.extend([s + r] * len(cols)); J.extend(cols.tolist()); S.extend(vals.tolist())
    return np.array(I), np.array(J), np.array(S)

def main():
    t0 = time.time()
    n = pd.read_pickle(f'{BASE}/work/state/s2_normalized.pkl').reset_index(drop=True)
    ids = n.id.values
    res = {}
    # (a) title words (drop very common tokens via max_df)
    tv = TfidfVectorizer(token_pattern=r'[a-z0-9]{2,}', sublinear_tf=True, max_df=0.02, min_df=1)
    Xt = tv.fit_transform(n.title_k.fillna(''))
    res['title'] = knn(Xt, K_TITLE, MIN_SIM['title'])
    print('title', len(res['title'][0]), time.time() - t0)
    # (b) title char 4-grams (robust to typos / merged words) on titles with >=15 chars
    cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 4), sublinear_tf=True, max_df=0.01, min_df=2)
    Xc = cv.fit_transform(n.title_k.fillna(''))
    res['char'] = knn(Xc, K_CHAR, MIN_SIM['char'])
    print('char', len(res['char'][0]), time.time() - t0)
    # (c) author surnames (+ first-initial surname pairs)
    def akey(L):
        out = []
        for k in L:
            p = k.split()
            if p:
                out.append(p[-1])
                out.append(p[0][0] + '_' + p[-1])
        return ' '.join(out)
    av = TfidfVectorizer(token_pattern=r'\S+', sublinear_tf=True, max_df=0.01, min_df=2)
    Xa = av.fit_transform(n.auth_keys.map(akey))
    res['auth'] = knn(Xa, K_AUTH, MIN_SIM['auth'])
    print('auth', len(res['auth'][0]), time.time() - t0)

    frames = []
    for m, (I, J, S) in res.items():
        a, b = np.minimum(I, J), np.maximum(I, J)
        frames.append(pd.DataFrame({'i': a, 'j': b, m: S}).groupby(['i', 'j'], as_index=False).max())
    c = frames[0]
    for f in frames[1:]:
        c = c.merge(f, on=['i', 'j'], how='outer')
    c['id1'] = ids[c.i]; c['id2'] = ids[c.j]
    c['s1'] = n.source.values[c.i]; c['s2'] = n.source.values[c.j]
    c.to_pickle(f'{BASE}/work/state/s3_candidates.pkl')
    # diagnostics
    cross = c[c.s1 != c.s2]
    d = {'n_pairs_total': int(len(c)), 'n_cross_source': int(len(cross)),
         'by_source_pair': cross.groupby(['s1', 's2']).size().rename('n').reset_index().astype(str).values.tolist(),
         'by_method': {m: int(c[m].notna().sum()) for m in res},
         'unique_to_method': {m: int((c[m].notna() & c[[x for x in res if x != m]].isna().all(axis=1)).sum()) for m in res},
         'same_source_pairs': int((c.s1 == c.s2).sum())}
    N = n.source.value_counts()
    full = N['crossref'] * N['dblp'] + N['crossref'] * N['open_alex'] + N['dblp'] * N['open_alex']
    d['reduction_ratio_cross'] = 1 - len(cross) / full
    deg = pd.concat([cross.id1, cross.id2]).value_counts()
    d['records_without_cross_candidate'] = int(len(n) - len(deg))
    d['records_without_cross_candidate_by_source'] = n[~n.id.isin(deg.index)].source.value_counts().to_dict()
    d['max_cross_degree'] = int(deg.max())
    d['secs'] = time.time() - t0
    print(json.dumps(d, indent=1))
    with open(f'{BASE}/work/diagnostics.jsonl', 'a') as f:
        f.write(json.dumps({'stage': 's3_blocking', 'input': 'work/state/s2_normalized.pkl',
                            'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), **d}) + '\n')

if __name__ == '__main__':
    main()
