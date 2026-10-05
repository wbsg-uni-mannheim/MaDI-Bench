"""Stage 3: blocking = union of (a) exact normalized title, (b) title char-ngram TF-IDF top-k,
(c) author-surname TF-IDF top-k. Per source pair; output all candidate pairs."""
import pandas as pd, numpy as np, json, itertools
from sklearn.feature_extraction.text import TfidfVectorizer
u = pd.read_pickle('work/state/s2_normalized.pkl')
SRC = ['crossref', 'dblp', 'open_alex']
K_TITLE, MIN_TITLE = 10, 0.35
K_AUTH, MIN_AUTH = 5, 0.5

def topk_pairs(A, B, ia, ib, k, thr, chunk=4000):
    out = []
    BT = B.T.tocsr()
    for s in range(0, A.shape[0], chunk):
        S = (A[s:s+chunk] @ BT).toarray()
        kk = min(k, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        val = np.take_along_axis(S, idx, axis=1)
        rr, cc = np.nonzero(val >= thr)
        out.extend(zip(ia[s + rr], ib[idx[rr, cc]]))
    return out

title_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, sublinear_tf=True, dtype=np.float32)
Xt = title_vec.fit_transform(u.title_key)
auth_txt = u.author_keys.map(lambda l: ' '.join(sorted(set(l))))
auth_vec = TfidfVectorizer(token_pattern=r'\S+', min_df=1, sublinear_tf=True, dtype=np.float32)
Xa = auth_vec.fit_transform(auth_txt)
ids = u.id.values
cands, stats = {}, []
for s1, s2 in itertools.combinations(SRC, 2):
    m1 = np.where(u.source == s1)[0]; m2 = np.where(u.source == s2)[0]
    P = {}
    # (a) exact title key
    a = u.iloc[m1][['id', 'title_key']]; b = u.iloc[m2][['id', 'title_key']]
    a = a[a.title_key.str.len() >= 3]; b = b[b.title_key.str.len() >= 3]
    ex = a.merge(b, on='title_key')
    for x, y in zip(ex.id_x, ex.id_y): P.setdefault((x, y), set()).add('exact')
    # (b) title tfidf both directions
    for x, y in topk_pairs(Xt[m1], Xt[m2], ids[m1], ids[m2], K_TITLE, MIN_TITLE): P.setdefault((x, y), set()).add('title_nn')
    for y, x in topk_pairs(Xt[m2], Xt[m1], ids[m2], ids[m1], K_TITLE, MIN_TITLE): P.setdefault((x, y), set()).add('title_nn')
    # (c) authors tfidf (only records with authors)
    h1 = m1[auth_txt.values[m1] != '']; h2 = m2[auth_txt.values[m2] != '']
    for x, y in topk_pairs(Xa[h1], Xa[h2], ids[h1], ids[h2], K_AUTH, MIN_AUTH): P.setdefault((x, y), set()).add('author_nn')
    for y, x in topk_pairs(Xa[h2], Xa[h1], ids[h2], ids[h1], K_AUTH, MIN_AUTH): P.setdefault((x, y), set()).add('author_nn')
    cands.update(P)
    meth = pd.Series([ '+'.join(sorted(v)) for v in P.values()]).value_counts().to_dict()
    ids1 = {p[0] for p in P}; ids2 = {p[1] for p in P}
    stats.append(dict(pair=f'{s1}-{s2}', candidates=len(P), full=len(m1) * len(m2), reduction=1 - len(P) / (len(m1) * len(m2)),
                      zero_cand_1=int(len(m1) - len(ids1)), zero_cand_2=int(len(m2) - len(ids2)), methods=meth))
    print(stats[-1])
c = pd.DataFrame([(a, b, '+'.join(sorted(v))) for (a, b), v in cands.items()], columns=['id1', 'id2', 'methods'])
c.to_pickle('work/state/s3_candidates.pkl')
json.dump(stats, open('work/state/s3_blocking_stats.json', 'w'), indent=1)
