"""Stage 4b: direct rule-based match decisions over the pairwise evidence (no training, no labels)."""
import pandas as pd, numpy as np, os, json, time, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = dict(T_HIGH=0.85, T_MID=0.6, A_WEAK=0.34, A_STRONG=0.75, A_PAGES=0.9)

def decide(c, n):
    ty = n.set_index(n.index).type_n.values
    A = np.fmax(c.a_min.values, c.a_tok.values)          # NaN only if both NaN
    A = pd.Series(A, index=c.index)
    short = (c[['len1', 'len2']].min(axis=1) <= 3)
    T = c.t_max.fillna(0)
    ydiff = c.y_diff
    pages_conf = ((c.fi_eq == 0) & (c.la_eq == 0)) | ((c.vo_eq == 0) & (c.fi_eq == 0)) | ((c.vo_eq == 0) & (c.is_eq == 0))
    strong_same = (T >= 0.9) & ((A >= P['A_STRONG']) | (A.isna() & (c.j_sim >= 0.8)))
    hard = (ydiff >= 3) | ((ydiff == 2) & ~strong_same) | pages_conf
    # commentary / erratum titles embed the commented paper's title: never the same publication
    tk = n.title_k.values
    pre = re.compile(r'^(comment(s)? on|reply to|response to|rejoinder|erratum|corrigendum|correction to|retraction|'
                     r'retracted|author correction|publisher correction|discussion of|editorial)\b')
    m1 = np.array([bool(pre.match(t)) for t in tk])
    meta_conf = pd.Series(m1[c.i.values] != m1[c.j.values], index=c.index)
    # differing numbers in titles ('WMT 2018' vs 'WMT 2019', 'Task 9' vs 'Task 12') mark different publications
    def numset(t):
        x = set(re.findall(r'\b\d+\b', t)) | set(re.findall(r'\b(ii|iii|iv|vi|vii|viii)\b', t))
        x |= {'part-' + p for p in re.findall(r'\bpart\s+([ivx]+|\d+)\b', t)}
        return frozenset(x)
    nums = [numset(t) for t in tk]
    num_conf = pd.Series([bool(nums[i]) and bool(nums[j]) and nums[i] != nums[j] for i, j in zip(c.i.values, c.j.values)],
                         index=c.index)
    venue_conf = (c.vo_eq == 0) & (c.j_sim < 0.5)
    hard = hard | meta_conf | num_conf | venue_conf
    support = ((c.vo_eq == 1).astype(int) + (c.fi_eq == 1).astype(int) + (c.j_sim >= 0.8).astype(int)
               + (ydiff == 0).astype(int) + (c.la_eq == 1).astype(int))
    # explicit type contradiction between the two sources that use dblp-style types (conference vs journal version)
    t1, t2 = ty[c.i.values], ty[c.j.values]
    s1, s2 = c.s1.values, c.s2.values
    reliable = np.isin(s1, ['dblp', 'crossref']) & np.isin(s2, ['dblp', 'crossref'])
    type_conf = pd.Series(reliable & (t1 != '') & (t2 != '') & (t1 != t2)
                          & np.isin(t1, ['article', 'inproceedings']) & np.isin(t2, ['article', 'inproceedings']),
                          index=c.index)
    y_ok = ydiff.isna() | (ydiff <= 1)
    Anan = A.isna()
    r1 = (~short) & (T >= P['T_HIGH']) & ~hard & (Anan | (A >= P['A_WEAK']) | ((support >= 2) & (ydiff.fillna(0) == 0)))
    contained = c.t_min >= 0.99     # one title (minus stopwords) contained in the other: truncation / dropped words
    r2 = (~short) & (T >= P['T_MID']) & (T < P['T_HIGH']) & ~hard & y_ok & (
        ((A >= P['A_STRONG']) & (contained | ((T >= 0.75) & (support >= 1))))
        | (Anan & (T >= 0.75) & (support >= 2)))
    r3 = ((c.t_exact == 1) | (c.t_colon == 1)) & ~hard & y_ok & (
        (A >= P['A_STRONG']) | (Anan & (support >= 2)))
    # long title fully contained in the other (garbled/markup tail) + same authors + corroboration
    r5 = (T < P['T_MID']) & (c[['len1', 'len2']].min(axis=1) >= 5) & (c.t_min >= 0.99) & (A >= P['A_PAGES']) \
        & (support >= 1) & y_ok & ~hard
    r4 = (T < P['T_MID']) & (A >= P['A_PAGES']) & (c.fi_eq == 1) & ((c.la_eq == 1) | (c.vo_eq == 1)) & y_ok & ~hard
    acc = (r1 | r2 | r3 | r4 | r5) & ~(type_conf & (ydiff.fillna(0) >= 1))
    # same-source pairs: a bibliographic source normally lists a publication once, so a within-source duplicate
    # needs stronger evidence (strict author agreement, near-identical title, no year/type contradiction)
    same = pd.Series(s1 == s2, index=c.index)
    tconf_same = pd.Series((t1 != '') & (t2 != '') & (t1 != t2), index=c.index)
    same_ok = (((A >= 0.85) | (Anan & (ydiff == 0) & (T >= 0.95)))
               & ((T >= 0.9) | ((c.t_min >= 0.99) & (T >= 0.75)))
               & (ydiff.isna() | (ydiff == 0)) & ~tconf_same & ~hard)
    acc = acc & (~same | same_ok)
    rule = np.select([r1, r2, r3, r4, r5], ['r1', 'r2', 'r3', 'r4', 'r5'], '')
    score = 0.5 * np.where(short & ((c.t_exact == 1) | (c.t_colon == 1)), 0.95, T) + 0.4 * A.fillna(0.5) + 0.02 * support
    score = score - 0.1 * type_conf.astype(float)
    return acc, rule, score, A, hard, type_conf

def main():
    n = pd.read_pickle(f'{BASE}/work/state/s2_normalized.pkl').reset_index(drop=True)
    c = pd.read_pickle(f'{BASE}/work/state/s4_features.pkl')
    acc, rule, score, A, hard, type_conf = decide(c, n)
    c['A'] = A; c['hard'] = hard; c['type_conf'] = type_conf
    c['accept'] = acc; c['rule'] = rule; c['score'] = np.round(score, 4)
    c.to_pickle(f'{BASE}/work/state/s4_scored.pkl')
    m = c[c.accept]
    d = {'accepted': int(len(m)), 'cross': int((m.s1 != m.s2).sum()), 'same_source': int((m.s1 == m.s2).sum()),
         'by_rule': m.rule.value_counts().to_dict(),
         'by_pair': m.groupby(['s1', 's2']).size().rename('n').reset_index().astype(str).values.tolist(),
         'rejected_by_typeconf': int(((rule != '') & ~acc).sum()), 'params': P}
    print(json.dumps(d, indent=1))
    with open(f'{BASE}/work/diagnostics.jsonl', 'a') as f:
        f.write(json.dumps({'stage': 's4_match', 'input': 'work/state/s4_features.pkl',
                            'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), **d}) + '\n')

if __name__ == '__main__':
    main()
