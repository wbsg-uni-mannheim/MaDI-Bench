"""Stage 7: write submission files from the final state artifacts."""
import pandas as pd, json, os
os.makedirs('submission/blocking', exist_ok=True)
f = pd.read_pickle('work/state/s6_fused.pkl')
f.to_csv('submission/fused.csv', index=False)
mem = pd.read_pickle('work/state/s5_membership.pkl')
mem.to_csv('submission/membership.csv', index=False)
cor = pd.read_pickle('work/state/s5_correspondences.pkl')
# transitive pairs (both matched to a common third record) that were never scored directly get the min score of the path
cor['score'] = cor.score.fillna(-1)
cor[['id1', 'id2', 'score']].to_csv('submission/correspondences.csv', index=False)
cand = pd.read_pickle('work/state/s3_candidates.pkl')[['id1', 'id2']]
# add transitive-closure pairs of the final clusters (documented as a separate candidate source)
key = set(zip(cand.id1, cand.id2)) | set(zip(cand.id2, cand.id1))
extra = cor[[(a, b) not in key for a, b in zip(cor.id1, cor.id2)]][['id1', 'id2']]
pd.concat([cand, extra]).to_csv('submission/blocking/candidates.csv', index=False)
print('fused', len(f), 'membership', len(mem), 'corr', len(cor), 'cand', len(cand) + len(extra), 'transitive extra', len(extra))
