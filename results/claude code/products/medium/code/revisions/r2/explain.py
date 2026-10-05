import sys, os; sys.path.insert(0, 'work')
import s4_match as M
for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    print(a, '|', M.recs[a].title[:70]); print(b, '|', M.recs[b].title[:70])
    print(M.evidence(a, b), M.feats[a]['mtok'], M.feats[b]['mtok'], M.feats[a]['codes'], M.feats[b]['codes'])
