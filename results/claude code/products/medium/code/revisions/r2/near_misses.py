import pandas as pd, sys
n = pd.read_pickle('work/state/normalized.pkl').set_index('id')
cl = pd.read_csv('work/state/clusters.csv').set_index('id').cluster
s = pd.read_csv('work/state/scored.csv', keep_default_na=False)
s = s[(s.id1.map(cl) != s.id2.map(cl))]
s = s[s.same_source == False] if s.same_source.dtype == bool else s[s.same_source == 'False']
kind = sys.argv[1]
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
if kind == 'rej':   # high-scoring but rejected by a conflict
    x = s[(s.score >= lo) & ((s.hard != '') | (s.soft != ''))]
    x = x[x.hard != 'capacity']
else:               # no conflict, but under threshold
    x = s[(s.hard == '') & (s.soft == '') & (s.score < 0.6) & (s.score >= lo)]
x = x.sort_values('score', ascending=False)
for r in x.itertuples():
    print(f'{r.score:.2f} {r.hard or r.soft:12s} {r.id1[9:]:10s} {n.loc[r.id1,"title"][:65]:65s} || {r.id2[9:]:10s} {n.loc[r.id2,"title"][:65]}')
