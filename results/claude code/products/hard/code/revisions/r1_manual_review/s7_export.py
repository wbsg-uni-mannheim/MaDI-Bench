"""Stage 7: export submission files from the final s5 membership (+ s6 fused table).
Candidate set = automatic blocking (s3) U all pairs inside each reviewed spec block of
work/manual/review_snapshot_v1.txt (every such pair was inspected) U pairs of explicit manual group lines."""
import pandas as pd, itertools, json
m = pd.read_csv('work/state/s5_membership.csv', dtype=str)
s3 = pd.read_csv('work/state/s3_candidates_all.csv', dtype=str)
src = dict(zip(m.record_id, m.source))
full = {x.replace('products_', ''): x for x in m.record_id}
def norm(x, y): return (x, y) if x < y else (y, x)
cand = {norm(x, y): 'auto' for x, y in zip(s3.id1, s3.id2)}
snap = open('work/manual/review_snapshot_v1.txt').read().split('\n'); blocks = {}; cur = None
for l in snap:
    if l.startswith('### '): cur = l[4:].strip(); blocks[cur] = []
    elif l.strip(): blocks[cur].append(full[l.split()[1]])
for b in blocks.values():
    for x, y in itertools.combinations(b, 2): cand.setdefault(norm(x, y), 'review_block')
for l in open('work/manual/decisions.txt').read().split('\n'):
    if l.strip() and not l.startswith('#'):
        for x, y in itertools.combinations([full[t] for t in l.split()], 2): cand.setdefault(norm(x, y), 'manual_link')
cc = [(x, y, k) for (x, y), k in cand.items() if src[x] != src[y]]
pd.DataFrame([(x, y) for x, y, _ in sorted(cc)], columns=['id1', 'id2']).to_csv('submission/blocking/candidates.csv', index=False)
corr = []
for cid, g in m.groupby('cluster_id'):
    for x, y in itertools.combinations(sorted(g.record_id), 2):
        if src[x] != src[y]: corr.append((x, y, 1.0))
pd.DataFrame(corr, columns=['id1', 'id2', 'score']).to_csv('submission/correspondences.csv', index=False)
m[['record_id', 'source', 'cluster_id']].to_csv('submission/membership.csv', index=False)
cs = set((x, y) for x, y, _ in cc)
st = dict(candidates_cross=len(cc), by_origin=pd.Series([k for _, _, k in cc]).value_counts().to_dict(),
          correspondences=len(corr), corr_in_candidates=sum(norm(x, y) in cs for x, y, _ in corr))
print(json.dumps(st)); json.dump(st, open('work/state/s7_stats.json', 'w'))
