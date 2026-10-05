"""Stage 5: graph clustering of accepted pairs with identity constraints.
Constraint: a cluster may not contain two *native* (non-injected) records of dbpedia or forbes (distinct URIs = distinct entities).
Violating components are split by removing weakest edges (lowest score first) until constraints hold."""
import os, sys, json, time, collections
import pandas as pd, numpy as np, networkx as nx
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def violates(nodes, R):
    c = collections.Counter()
    for n in nodes:
        if R.source.iat[n] in ('dbpedia', 'forbes') and not R.native_dup.iat[n]: c[R.source.iat[n]] += 1
    return any(v > 1 for v in c.values())
def main():
    R = pd.read_pickle(f'{ROOT}/work/state/records.pkl')
    P = pd.read_pickle(f'{ROOT}/work/state/pair_decisions.pkl')
    A = P[P.accept]
    G = nx.Graph(); G.add_nodes_from(range(len(R)))
    for r in A.itertuples(): G.add_edge(r.i, r.j, score=r.score, rule=r.rule)
    removed = []
    changed = True
    while changed:
        changed = False
        for comp in list(nx.connected_components(G)):
            if len(comp) > 1 and violates(comp, R):
                sub = G.subgraph(comp)
                # remove the weakest edge that lies on a path between two conflicting natives: approximate by global weakest edge in component
                e = min(sub.edges(data=True), key=lambda x: (x[2]['score'], x[2]['rule'] != 'forbes_url_link'))
                G.remove_edge(e[0], e[1]); removed.append((R.rid.iat[e[0]], R.rid.iat[e[1]], e[2]['score'], e[2]['rule']))
                changed = True
    comps = sorted(nx.connected_components(G), key=lambda c: min(c))
    cid = np.empty(len(R), dtype=object)
    for k, comp in enumerate(comps):
        for n in comp: cid[n] = f'E{k:05d}'
    R['cluster'] = cid
    R.to_pickle(f'{ROOT}/work/state/records_clustered.pkl')
    pd.DataFrame(removed, columns=['id1', 'id2', 'score', 'rule']).to_csv(f'{ROOT}/work/state/rejected_edges.csv', index=False)
    # final edge list = accepted edges still in graph
    kept = [(R.rid.iat[a], R.rid.iat[b], d['score'], d['rule']) for a, b, d in G.edges(data=True)]
    pd.DataFrame(kept, columns=['id1', 'id2', 'score', 'rule']).to_pickle(f'{ROOT}/work/state/kept_edges.pkl')
    sizes = R.groupby('cluster').size()
    nsrc = R.groupby('cluster').source.nunique()
    diag = {'ts': time.time(), 'stage': 'clustering', 'inputs': 'work/state/pair_decisions.pkl', 'records': len(R), 'clusters': int(len(sizes)),
            'singletons': int((sizes == 1).sum()), 'singleton_share': float((sizes == 1).mean()), 'max_size': int(sizes.max()),
            'size_dist': {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
            'sources_per_cluster': {int(k): int(v) for k, v in nsrc.value_counts().sort_index().items()},
            'edges_accepted': int(len(A)), 'edges_removed_by_constraints': len(removed)}
    print(json.dumps(diag, indent=1))
    with open(f'{ROOT}/work/diagnostics.jsonl', 'a') as fh: fh.write(json.dumps(diag) + '\n')
if __name__ == '__main__':
    main()
