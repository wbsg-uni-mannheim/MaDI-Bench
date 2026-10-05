"""Stage 1: schema matching. All four sources share the target attribute names except the URL column."""
import json, pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
schema = json.load(open('task/input/schemamatching/target_schema.json'))
targets = [k for k in schema['properties']]
rows, mapping = [], {}
for i in range(1, 5):
    src = f'products_{i}'
    cols = pd.read_csv(f'task/input/data/{src}.csv', nrows=5).columns
    m = {}
    for c in cols:
        if c in targets:
            m[c] = c; score = 1.0
        elif c.lower() in ('product_url', 'producturl', 'link'):
            m[c] = 'url'; score = 0.95   # URL of offer page; examples are http(s) links
        else:
            continue
        rows.append((src, c, 'target', m[c], score))
    mapping[src] = m
pd.DataFrame(rows, columns=['source_dataset','source_column','target_dataset','target_column','score']).to_csv('work/state/sm_mapping.csv', index=False)
json.dump(mapping, open('work/state/sm_mapping.json','w'), indent=1)
print({s: len(m) for s, m in mapping.items()}, 'targets without contributors:', set(targets) - set(v for m in mapping.values() for v in m.values()))
