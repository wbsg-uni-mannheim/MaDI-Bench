"""Stage 4: pairwise features + rule-based decisions (no training)."""
import os, sys, json, time
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import pair_features, set_idf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def main():
    R = pd.read_pickle(f'{ROOT}/work/state/records.pkl')
    P = pd.read_pickle(f'{ROOT}/work/state/candidates.pkl')
    E = np.load(f'{ROOT}/work/state/E_primary.npy')
    set_idf(list(R.nkey_core) + [t for t in R.tkey_core if t])
    recs = R.to_dict('records')
    feats = [pair_features(recs[i], recs[j]) for i, j in zip(P.i, P.j)]
    F = pd.DataFrame(feats)
    P = pd.concat([P.reset_index(drop=True), F], axis=1)
    P['emb'] = np.einsum('ij,ij->i', E[P.i.values], E[P.j.values])
    P.to_pickle(f'{ROOT}/work/state/pair_features.pkl')
    print(P.describe().T.to_string())
if __name__ == '__main__' and len(sys.argv) == 1:
    main()
    import subprocess

# ---------------- decision rules ----------------
GENERIC_INFO = 8.5   # IDF mass below this = name made only of common tokens (e.g. 'Industries', 'Motors')
def decide(r):
    """returns (accept:bool, score:float, rule:str). Missing attributes are unknown (0), never agreement."""
    if r.s1 == r.s2 and r.s1 != 'fullcontact' and not (r.d1 or r.d2):
        return False, 0.0, 'same_source_distinct_native_ids'
    if 'link' in r.methods:
        return True, 1.0, 'forbes_url_link'
    pos = sum(1 for k in ('year', 'city', 'industry', 'assets', 'revenue', 'people') if getattr(r, k) == 1)
    strong_neg = (r.year == -1) + (r.city == -1) + (r.people == -1)
    money_neg = (r.assets == -1) + (r.revenue == -1)
    info = min(r.info_a, r.info_b)
    ns = r.ns
    score = ns + 0.05 * pos - 0.05 * (strong_neg + money_neg) + 0.05 * r.country
    if ns >= 0.97:
        if info >= GENERIC_INFO:
            if r.country == -1 and strong_neg >= 1 and pos == 0: return False, score, 'exact_name_but_country+attr_conflict'
            if r.country == -1 and not ('forbes' in (r.s1, r.s2)) and pos == 0: return False, score, 'exact_name_country_conflict_no_support'
            return True, score, 'exact_name'
        if r.country == 1 and pos >= 1 and strong_neg == 0: return True, score, 'generic_exact_name+support'
        return False, score, 'generic_exact_name_unsupported'
    if ns >= 0.85:
        if r.country != -1 and pos >= 1 and strong_neg == 0 and money_neg == 0: return True, score, 'close_name+support'
        if r.country == 1 and strong_neg == 0 and money_neg == 0 and r.emb >= 0.8: return True, score, 'close_name+country+emb'
        return False, score, 'close_name_unsupported'
    n_extra = len(r.extra.split()) if isinstance(r.extra, str) and r.extra else 0
    if r.covmax >= 0.999 and ns >= 0.6 and n_extra <= 1 and r.emb >= 0.75 and r.country != -1 and strong_neg == 0 and money_neg == 0 \
            and (r.country == 1 or pos >= 1 or 'forbes' in (r.s1, r.s2)):
        return True, score, 'contained_name+support'
    if ns >= 0.7:
        if r.country == 1 and pos >= 2 and strong_neg == 0 and money_neg == 0: return True, score, 'partial_name+2support'
        return False, score, 'partial_name_unsupported'
    return False, score, 'low_name'
def run_decisions():
    P = pd.read_pickle(f'{ROOT}/work/state/pair_features.pkl')
    R = pd.read_pickle(f'{ROOT}/work/state/records.pkl')
    P['d1'] = R.native_dup.values[P.i]; P['d2'] = R.native_dup.values[P.j]
    out = [decide(r) for r in P.itertuples(index=False)]
    P['accept'] = [o[0] for o in out]; P['score'] = [o[1] for o in out]; P['rule'] = [o[2] for o in out]
    P.to_pickle(f'{ROOT}/work/state/pair_decisions.pkl')
    print(P.groupby(['rule', 'accept']).size())
    print(P[P.accept].groupby(['s1', 's2']).size())
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'decide':
    run_decisions()
