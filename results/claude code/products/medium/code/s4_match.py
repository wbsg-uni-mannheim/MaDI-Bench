"""Stage 4: pairwise entity matching over blocking candidates with interpretable evidence.
No training; hand-set rule weights in work/match_config.json (rationale in report). Writes work/state/scored.csv
and work/state/features.pkl (per-record matching features reused by s5)."""
import pandas as pd, numpy as np, os, re, json, pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pairlib import key_agreement, lev, mn_noise_equal, hard_conflict, soft_conflict, partno_conflict, strong_shared, near, model_tokens, title_words
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
n = pd.read_pickle(f'{W}/state/normalized.pkl').set_index('id', drop=False)
c = pd.read_csv(f'{W}/state/candidates.csv')
CFG = json.load(open(f'{W}/match_config.json'))

def wtok(s):
    s = s.lower().replace('™', ' ').replace('®', ' ')
    s = re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])', ' ', s)
    return ' '.join(re.findall(r'[a-z]+|\d+(?:\.\d+)?', s))
text = (n.title + ' ' + n.model).map(wtok)
X = TfidfVectorizer(token_pattern=r'[^ ]+', sublinear_tf=True).fit_transform(text)
Xt = TfidfVectorizer(token_pattern=r'[^ ]+', sublinear_tf=True).fit_transform(n.title.map(wtok))   # title only
pos = {k: i for i, k in enumerate(n.index)}
code_df = Counter(x for cs in n.n_codes for x in set(cs))
recs = {i: r for i, r in n.iterrows()}
feats = {i: dict(mtok=model_tokens(r), words=title_words(r),
                 codes={x for x in r.n_codes if code_df[x] <= CFG['max_code_df']}) for i, r in recs.items()}

def evidence(a, b):
    ra, rb, fa, fb = recs[a], recs[b], feats[a], feats[b]
    ev = {'hard': hard_conflict(ra, rb) or ''}
    cos = float(X[pos[a]].multiply(X[pos[b]]).sum())
    tcos = float(Xt[pos[a]].multiply(Xt[pos[b]]).sum())
    strong = strong_shared(fa['codes'], fb['codes'])
    shared = (fa['codes'] & fb['codes']) | strong
    mn = None
    if ra.n_mn and rb.n_mn:
        mn = 'same' if (near(ra.n_mn, rb.n_mn) or ra.n_mn in rb.n_mn or rb.n_mn in ra.n_mn) else 'diff'
    soft = soft_conflict(ra, rb, fa, fb) or ''
    # differing part numbers contradict identity, except between near-identical titles (then a one-off
    # corrupted model number is the likelier explanation; OCR-style noise is pervasive in these sources)
    if not soft and not strong and cos < CFG['partno_ignore_cos'] and partno_conflict(fa['codes'], fb['codes']): soft = 'partno'
    key = key_agreement(ra, rb, fa, fb)
    s = max(cos, tcos) + (CFG['w_key'] if key else 0) + (CFG['w_code'] if shared else 0) + (CFG['w_mn_same'] if mn == 'same' else 0) \
        - (CFG['w_mn_diff'] if (mn == 'diff' and not shared) else 0)
    # a shared exact part code overrides soft (model-line) contradictions, never hard ones
    if soft and strong and soft != 'gpu_edition': soft = ''
    if soft in ('model_tokens', 'line_or_model') and cos >= CFG['partno_ignore_cos']: soft = ''   # see partno note
    if soft == 'brand' and ra.n_mn and rb.n_mn and ra.n_mn == rb.n_mn and len(ra.n_mn) >= 7: soft = ''
    # records without any recognised brand, capacity or shared identifier (generic marketplace listings):
    # only near-identical titles are accepted
    if not ra.n_real_brand and not rb.n_real_brand and not shared and mn != 'same' and max(cos, tcos) < CFG['generic_min_cos']:
        soft = soft or 'generic_low_title_sim'
    # both records without a recognised brand (synthetic catalogue items share model numbers across
    # look-alike products while their own copies carry OCR-corrupted model numbers): require a near-equal
    # model number (<=2 edits) when both have one, and similar titles
    if not ra.n_real_brand and not rb.n_real_brand:
        if ra.n_mn and rb.n_mn and not mn_noise_equal(ra.n_mn, rb.n_mn): soft = soft or 'syn_model_number'
        if max(cos, tcos) < CFG['syn_min_cos']: soft = soft or 'syn_title'
        if soft in ('partno', 'model_tokens', 'line_or_model') and ra.n_mn and rb.n_mn and mn_noise_equal(ra.n_mn, rb.n_mn) \
                and max(cos, tcos) >= CFG['syn_min_cos']: soft = ''
    # fictional-brand copies with identical titles and identical long model numbers: brand-field noise, not a conflict
    if ev['hard'] == 'brand_head' and strong and max(cos, tcos) >= CFG['partno_ignore_cos']: ev['hard'] = ''
    ev.update(key=key, cos=round(cos, 3), tcos=round(tcos, 3), shared='|'.join(sorted(shared)), strong='|'.join(sorted(strong)), mn=mn or '', soft=soft, score=round(s, 3))
    return ev

if __name__ == '__main__':
    out = []
    for a, b, why in c.itertuples(index=False):
        ev = evidence(a, b); ev.update(id1=a, id2=b, why=why); out.append(ev)
    s = pd.DataFrame(out)
    s['same_source'] = s.id1.map(n.source) == s.id2.map(n.source)
    thr = np.where(s.same_source, CFG['threshold_same_source'], CFG['threshold'])
    s['accept'] = (s.hard == '') & (s.soft == '') & (s.score >= thr)
    s.to_csv(f'{W}/state/scored.csv', index=False)
    print(int(s.accept.sum()), 'accepted of', len(s))
    print('hard', s.hard.value_counts().to_dict()); print('soft', s.soft.value_counts().to_dict())
