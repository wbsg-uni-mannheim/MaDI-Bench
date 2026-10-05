"""Stage 4: pairwise evidence + rule-based decisions on the blocking candidates."""
import os, json, re
import pandas as pd, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairfeat import pair_features, code_doc_freq, set_word_df
from sklearn.feature_extraction.text import TfidfVectorizer
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
d = pd.read_pickle('work/state/norm.pkl')
c = pd.read_pickle('work/state/candidates.pkl')
tfw = TfidfVectorizer(analyzer='word', token_pattern=r'[a-z0-9.]+', sublinear_tf=True).fit(d.title_n)
W = tfw.transform(d.title_n)
tfc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True).fit(d.title_n)
C = tfc.transform(d.title_n)
i1, i2 = c.i1.values, c.i2.values
c['cos_w'] = np.asarray(W[i1].multiply(W[i2]).sum(1)).ravel()
c['cos_c'] = np.asarray(C[i1].multiply(C[i2]).sum(1)).ravel()
c['cos'] = (c.cos_w + c.cos_c) / 2
code_df = code_doc_freq(d)
set_word_df(d.title_n.tolist(), d.brand_key.tolist())
R = d.to_dict('records')
F = pd.DataFrame([pair_features(R[a], R[b], code_df) for a, b in zip(i1, i2)])
c = pd.concat([c.reset_index(drop=True), F], axis=1)
c['strong_id'] = c.mn_eq | (c.n_shared > 0)
# decision rules (see report): identifier evidence + some title agreement, or near-identical titles without contradictions
T_ID, T_TXT, T_TXT_SPEC = 0.25, 0.65, 0.50
c['rule'] = np.where(c.hard_conf, 'reject_conflict',
            np.where(c.mn_eq | ((c.n_shared > 0) & ~c.code_contra & (c.cos >= T_ID)), 'id',
            np.where(~c.code_contra & ~c.type_conf & ~c.vdiff & ~c.mtok_contra & ~c.iface_contra & ((c.cos >= T_TXT) | ((c.cos >= T_TXT_SPEC) & c.spec_agree & c.brand_agree & ~c.two_sided)), 'text', 'reject')))
# two records of the same source are only linked on an exact model-number match (sellers rarely list one product twice)
c.loc[c.same_src & ~c.mn_eq & c.rule.isin(['id', 'text']), 'rule'] = 'reject_same_source'
c['match'] = c.rule.isin(['id', 'text'])
c['score'] = (0.5 * c.strong_id + 0.5 * c.cos).round(4)
c.to_pickle('work/state/scored.pkl')
stats = {'rules': c.rule.value_counts().to_dict(), 'matches': int(c.match.sum()),
         'matches_same_source': int((c.match & c.same_src).sum())}
print(json.dumps(stats)); json.dump(stats, open('work/state/matching_stats.json', 'w'))
