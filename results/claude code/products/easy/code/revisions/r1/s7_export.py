"""Stage 7: write submission files from the saved state of the same run."""
import os, shutil, json
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
os.makedirs('submission/blocking', exist_ok=True)
shutil.copy('work/state/sm_mapping.csv', 'submission/sm_mapping.csv')
c = pd.read_pickle('work/state/candidates.pkl')
corr = pd.read_csv('work/state/correspondences.csv')
# the exported candidate set also contains the few within-cluster pairs that were joined only transitively
# (A-B and B-C were direct candidates, A-C was not), so every correspondence is a member of the candidate set
key = lambda a, b: (a, b) if a < b else (b, a)
cs = {key(a, b) for a, b in zip(c.id1, c.id2)}
extra = sorted({key(a, b) for a, b in zip(corr.id1, corr.id2)} - cs)
pd.DataFrame(sorted(cs) + extra, columns=['id1', 'id2']).to_csv('submission/blocking/candidates.csv', index=False)
json.dump({'blocking_pairs': len(cs), 'closure_pairs_added': len(extra)}, open('work/state/export_stats.json', 'w'))
corr.to_csv('submission/correspondences.csv', index=False)
pd.read_csv('work/state/membership.csv').to_csv('submission/membership.csv', index=False)
pd.read_csv('work/state/fused.csv', dtype=str, keep_default_na=False).to_csv('submission/fused.csv', index=False)
print('exported')
