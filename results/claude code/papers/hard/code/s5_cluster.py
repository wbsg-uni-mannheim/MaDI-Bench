"""Stage 5: constrained greedy clustering of accepted pairs.
Edges are merged in descending score order; a merge is refused when it would put two records of the
same source together that are not themselves directly matched (same-source duplicates need direct evidence).
This prevents weak bridges from chaining e.g. a conference paper and its journal extension."""
import pandas as pd, numpy as np, os, json, time, collections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    n = pd.read_pickle(f'{BASE}/work/state/s2_normalized.pkl')
    c = pd.read_pickle(f'{BASE}/work/state/s4_scored.pkl')
    m = c[c.accept].sort_values(['score', 'id1', 'id2'], ascending=[False, True, True])
    direct = set(zip(m.id1, m.id2)) | set(zip(m.id2, m.id1))
    parent = {i: i for i in n.id}
    members = {i: {s: [i]} for i, s in zip(n.id, n.source)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    kept, rejected = [], []
    for id1, id2, sc in zip(m.id1, m.id2, m.score):
        a, b = find(id1), find(id2)
        if a == b:
            kept.append((id1, id2, sc)); continue
        ok = True
        for s in set(members[a]) & set(members[b]):
            for x in members[a][s]:
                for y in members[b][s]:
                    if (x, y) not in direct:
                        ok = False; break
                if not ok: break
            if not ok: break
        if not ok:
            rejected.append((id1, id2, sc)); continue
        if sum(map(len, members[a].values())) < sum(map(len, members[b].values())):
            a, b = b, a
        parent[b] = a
        for s, L in members[b].items():
            members[a].setdefault(s, []).extend(L)
        del members[b]
        kept.append((id1, id2, sc))
    root = {i: find(i) for i in n.id}
    comp = collections.defaultdict(list)
    for i, r in root.items():
        comp[r].append(i)
    cid = {}
    for r, L in comp.items():
        name = 'E_' + sorted(L)[0]
        for i in L:
            cid[i] = name
    mem = pd.DataFrame({'record_id': n.id.values, 'source': n.source.values})
    mem['cluster_id'] = mem.record_id.map(cid)
    mem.to_pickle(f'{BASE}/work/state/s5_membership.pkl')
    # correspondences: all cross-source pairs inside final clusters that were accepted edges (kept), plus
    # implied cross-source pairs within a cluster (transitive closure) so that pairs agree with membership
    kept_df = pd.DataFrame(kept, columns=['id1', 'id2', 'score'])
    pairs = []
    sc = {(a, b): s for a, b, s in kept} | {(b, a): s for a, b, s in kept}
    for r, L in comp.items():
        if len(L) < 2:
            continue
        L = sorted(L)
        for x in range(len(L)):
            for y in range(x + 1, len(L)):
                if L[x].split('-')[0] != L[y].split('-')[0]:
                    pairs.append((L[x], L[y], sc.get((L[x], L[y]), np.nan)))
    corr = pd.DataFrame(pairs, columns=['id1', 'id2', 'score'])
    corr['implied'] = corr.score.isna()
    corr.to_pickle(f'{BASE}/work/state/s5_correspondences.pkl')
    pd.DataFrame(rejected, columns=['id1', 'id2', 'score']).to_csv(f'{BASE}/work/state/s5_rejected_edges.csv', index=False)
    # diagnostics
    sizes = mem.groupby('cluster_id').size()
    nsrc = mem.groupby('cluster_id').source.nunique()
    multi = mem.groupby(['cluster_id', 'source']).size()
    d = {'records': int(len(mem)), 'clusters': int(len(sizes)), 'singletons': int((sizes == 1).sum()),
         'singleton_share': round(float((sizes == 1).mean()), 4),
         'size_dist': {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
         'sources_per_cluster': {int(k): int(v) for k, v in nsrc.value_counts().sort_index().items()},
         'clusters_with_same_source_multiples': int((multi > 1).groupby(level=0).any().sum()),
         'accepted_edges': int(len(m)), 'kept_edges': len(kept), 'rejected_edges': len(rejected),
         'correspondences': int(len(corr)), 'implied_correspondences': int(corr.implied.sum()),
         'unmatched_by_source': mem[mem.cluster_id.map(sizes) == 1].source.value_counts().to_dict()}
    print(json.dumps(d, indent=1))
    with open(f'{BASE}/work/diagnostics.jsonl', 'a') as f:
        f.write(json.dumps({'stage': 's5_cluster', 'input': 'work/state/s4_scored.pkl',
                            'ts': time.strftime('%Y-%m-%dT%H:%M:%S'), **d}) + '\n')

if __name__ == '__main__':
    main()
