"""Stage 3: blocking. Union of (a) rare-token blocks on name keys, (b) embedding kNN over primary names,
(c) forbes url links (within-source duplicate pointer), (d) exact core-key equality. Includes within-source pairs
(sources contain injected duplicates)."""
import os, sys, collections, itertools, json, time
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed import embed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_DF = 40      # skip tokens appearing in more records than this (generic words)
KNN = 12
def primary_name(r):
    return r['title_clean'] or r['name_clean'] or ''
def main():
    R = pd.read_pickle(f'{ROOT}/work/state/records.pkl')
    R['pname'] = R.apply(primary_name, axis=1)
    n = len(R)
    pairs = collections.defaultdict(set)
    # (a) token blocks
    toks = []
    for _, r in R.iterrows():
        t = set(r['nkey_core'].split()) | set(r['tkey_core'].split()) if r['tkey_core'] else set(r['nkey_core'].split())
        toks.append({x for x in t if len(x) >= 2})
    inv = collections.defaultdict(list)
    for i, t in enumerate(toks):
        for x in t: inv[x].append(i)
    big = {k: len(v) for k, v in inv.items() if len(v) > MAX_DF}
    for k, v in inv.items():
        if len(v) > MAX_DF: continue
        for a, b in itertools.combinations(v, 2): pairs[(min(a, b), max(a, b))].add('token')
    # (b) embedding kNN on primary + raw names
    texts = R.pname.fillna('').tolist()
    raw = R.raw_name.fillna('').tolist()
    E1 = embed(texts); E2 = embed(raw)
    for E in (E1, E2):
        S = E1 @ E.T if E is E2 else E1 @ E1.T
        np.fill_diagonal(S, -1)
        idx = np.argpartition(-S, KNN, axis=1)[:, :KNN]
        for i in range(n):
            for j in idx[i]:
                if S[i, j] >= 0.5: pairs[(min(i, j), max(i, j))].add('emb')
    np.save(f'{ROOT}/work/state/E_primary.npy', E1); np.save(f'{ROOT}/work/state/E_raw.npy', E2)
    # (c) forbes links
    pos = {rid: i for i, rid in enumerate(R.rid)}
    for i, l in enumerate(R.link):
        if l and l in pos: j = pos[l]; pairs[(min(i, j), max(i, j))].add('link')
    # (d) exact core key
    kk = collections.defaultdict(list)
    for i, (a, b) in enumerate(zip(R.nkey_core, R.tkey_core)):
        for k in {a, b}:
            if k: kk[k].append(i)
    for k, v in kk.items():
        if len(v) <= 200:
            for a, b in itertools.combinations(v, 2): pairs[(min(a, b), max(a, b))].add('exact')
    P = pd.DataFrame([(a, b, '|'.join(sorted(m))) for (a, b), m in pairs.items() if a != b], columns=['i', 'j', 'methods'])
    P['id1'] = R.rid.values[P.i]; P['id2'] = R.rid.values[P.j]
    P['s1'] = R.source.values[P.i]; P['s2'] = R.source.values[P.j]
    P.to_pickle(f'{ROOT}/work/state/candidates.pkl')
    R.to_pickle(f'{ROOT}/work/state/records.pkl')
    # diagnostics
    tot = n * (n - 1) / 2
    diag = {'ts': time.time(), 'stage': 'blocking', 'inputs': 'work/state/records.pkl', 'n_records': n, 'candidates': len(P), 'reduction_ratio': 1 - len(P) / tot,
            'by_source_pair': P.groupby(['s1', 's2']).size().to_dict(), 'by_method': P.methods.value_counts().head(10).to_dict(),
            'skipped_tokens_over_maxdf': len(big)}
    deg = collections.Counter(list(P.i) + list(P.j))
    diag['zero_candidate_records'] = int(sum(1 for i in range(n) if deg[i] == 0))
    diag['max_candidates_per_record'] = max(deg.values())
    diag['by_source_pair'] = {f'{a}-{b}': int(v) for (a, b), v in diag['by_source_pair'].items()}
    print(json.dumps(diag, indent=1))
    with open(f'{ROOT}/work/diagnostics.jsonl', 'a') as fh: fh.write(json.dumps(diag) + '\n')
if __name__ == '__main__':
    main()
