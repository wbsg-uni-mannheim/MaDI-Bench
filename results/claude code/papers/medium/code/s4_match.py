"""Stage 4b: score candidates, accept score >= THRESH, enforce one-to-one per source pair (greedy by score)."""
import sys, pandas as pd, numpy as np
sys.path.insert(0, 'work')
from score import score
THRESH = 3.0
f = pd.read_pickle('work/state/s4_features.pkl')
f['score'] = score(f)
f['pair'] = f.id1.str.rsplit('-', n=1).str[0] + '~' + f.id2.str.rsplit('-', n=1).str[0]
f[['id1', 'id2', 'pair', 'score']].to_pickle('work/state/s4_scores.pkl')
acc = f[f.score >= THRESH].sort_values(['score', 'id1', 'id2'], ascending=[False, True, True])
used, keep = set(), []
for a, b, p, s in zip(acc.id1, acc.id2, acc.pair, acc.score):
    if (p, a) in used or (p, b) in used: continue
    used.add((p, a)); used.add((p, b)); keep.append((a, b, s))
m = pd.DataFrame(keep, columns=['id1', 'id2', 'score'])
m.to_pickle('work/state/s4_matches.pkl')
print('above threshold', len(acc), 'kept 1:1', len(m))
print(m.assign(p=m.id1.str.split('-').str[0] + '~' + m.id2.str.split('-').str[0]).p.value_counts())
