"""Stage 6: decisions + constrained clustering.
Accept gated pairs with score >= T. Ambiguous edges (an endpoint has another gated candidate
from the same other source within MARGIN) are flagged for diagnostics only; ties are broken
deterministically by score, then by the lower numeric native id (earlier catalogued record). Greedy merge by descending score; a merge
is refused if it would put two records of one source together (sources hold one row per release)
or if any explicitly scored cross pair between the two clusters is contradictory (score < CONTRA)."""
import pandas as pd, numpy as np, json, sys, datetime
T, MARGIN, CONTRA = float(sys.argv[1]) if len(sys.argv) > 1 else 3.5, 0.25, 2.5
F = pd.read_pickle("work/state/scored.pkl")
R = pd.read_pickle("work/state/records.pkl")
SRC = dict(zip(R.id, R.source))
G = F[F.gate].copy()
G["src1"] = G.id1.map(SRC); G["src2"] = G.id2.map(SRC)
# ambiguity: best alternative score for each endpoint towards the same other source
def alt_best(df, me, other):
    s = df.sort_values("score", ascending=False)
    top = s.groupby(me).score.apply(lambda x: list(x.values[:2]))
    return top
amb = np.zeros(len(G), bool)
for me, other in [("id1", "id2"), ("id2", "id1")]:
    grp = G.groupby([me, "src1" if other == "id1" else "src2"]).score
    second = grp.transform(lambda x: np.sort(x.values)[-2] if len(x) > 1 else -9)
    first = grp.transform("max")
    # an edge is ambiguous from this endpoint if it is (near-)best but a different candidate is within MARGIN
    amb |= ((G.score >= first - 1e-9) & (second >= G.score - MARGIN)).values | ((G.score < first - 1e-9) & (G.score >= first - MARGIN)).values
G["ambiguous"] = amb
G["accepted_pair"] = G.score >= T
num = lambda s: s.str.extract(r"(\d+)$")[0].astype(int)
G["n1"] = num(G.id1); G["n2"] = num(G.id2)
E = G[G.accepted_pair].sort_values(["score", "n1", "n2"], ascending=[False, True, True])
pairscore = {}
for a, b, s in zip(F.id1, F.id2, F.score): pairscore[(a, b)] = s; pairscore[(b, a)] = s
parent = {i: i for i in R.id}; members = {i: [i] for i in R.id}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
log = []
for a, b, s in zip(E.id1, E.id2, E.score):
    ra, rb = find(a), find(b)
    if ra == rb: log.append((a, b, s, "intra")); continue
    sa = {SRC[x] for x in members[ra]}; sb = {SRC[x] for x in members[rb]}
    if sa & sb: log.append((a, b, s, "source_conflict")); continue
    contra = [pairscore.get((x, y)) for x in members[ra] for y in members[rb] if (x, y) != (a, b) and (x, y) != (b, a)]
    if any(c is not None and c < CONTRA for c in contra): log.append((a, b, s, "contradiction")); continue
    parent[rb] = ra; members[ra] += members.pop(rb); log.append((a, b, s, "merged"))
L = pd.DataFrame(log, columns=["id1", "id2", "score", "decision"])
L.to_csv("work/state/edge_decisions.csv", index=False)
G.to_pickle("work/state/gated.pkl")
cl = {i: find(i) for i in R.id}
M = pd.DataFrame({"record_id": R.id, "source": R.source, "root": R.id.map(cl)})
M.to_pickle("work/state/clusters.pkl")
# correspondences = all within-cluster cross-source pairs
corr = []
for root, mem in M.groupby("root").record_id:
    mem = sorted(mem)
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            corr.append((mem[i], mem[j], round(pairscore.get((mem[i], mem[j]), np.nan), 4)))
C = pd.DataFrame(corr, columns=["id1", "id2", "score"])
C.to_pickle("work/state/corr.pkl")
# transitively implied pairs that were never blocking candidates are appended to the exported candidate set
cand = pd.read_csv("work/state/candidates.csv")[["id1", "id2"]]
cs = set(zip(cand.id1, cand.id2)) | set(zip(cand.id2, cand.id1))
extra = C[[(a, b) not in cs for a, b in zip(C.id1, C.id2)]][["id1", "id2"]]
pd.concat([cand, extra]).to_csv("submission/blocking/candidates.csv", index=False)
sz = M.groupby("root").size(); ns = M.groupby("root").source.nunique()
comp = M.groupby("root").source.apply(lambda x: "+".join(sorted(x)))
diag = dict(stage="cluster", T=T, margin=MARGIN, gated=len(G), ambiguous=int(G.ambiguous.sum()),
            above_T=int(G.accepted_pair.sum()), edge_decisions=L.decision.value_counts().to_dict(),
            clusters=len(sz), size_dist=sz.value_counts().to_dict(), composition=comp[sz > 1].value_counts().to_dict(),
            corr=len(C), implied_noncandidate_pairs=len(extra), corr_missing_score=int(C.score.isna().sum()),
            linked_share={s: round(float((M[M.source == s].root.map(sz) > 1).mean()), 4) for s in SRC.values() and ["discogs", "lastfm", "musicbrainz"]})
print(json.dumps(diag, indent=1))
diag.update(ts=datetime.datetime.now().isoformat(), inputs="work/state/scored.pkl")
open("work/diagnostics.jsonl", "a").write(json.dumps(diag) + "\n")
