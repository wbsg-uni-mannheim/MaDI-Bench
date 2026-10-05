"""Stage 7: export deliverables to submission/ from work/state (same revision)."""
import pandas as pd, os, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
os.makedirs('submission/blocking', exist_ok=True)
pd.read_csv('work/state/membership.csv').to_csv('submission/membership.csv', index=False)
pd.read_csv('work/state/correspondences.csv').to_csv('submission/correspondences.csv', index=False)
pd.read_csv('work/state/fused.csv', dtype={'assets': 'Int64', 'revenue': 'Int64'}).to_csv('submission/fused.csv', index=False)
c = pd.read_csv('work/state/candidates.csv')[['id1', 'id2']]
c.to_csv('submission/blocking/candidates.csv', index=False)
print('exported')
