"""Stage 1: schema mapping. All three sources use target-like column names; verified by
inspecting example values (see report). Unmapped: publisher, abstract_text, keywords."""
import pandas as pd, json, os
W=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(W)
schema=json.load(open(f'{R}/task/input/schemamatching/target_schema.json'))
targets=[t for t in schema['properties']]
rows=[]
for src in ['crossref','dblp','open_alex']:
    cols=pd.read_csv(f'{R}/task/input/data/{src}.csv',nrows=5).columns
    for c in cols:
        if c in targets:
            rows.append((src,c,'papers',c,1.0))
        else:
            print(src,'unmapped:',c)
os.makedirs(f'{R}/submission',exist_ok=True)
pd.DataFrame(rows,columns=['source_dataset','source_column','target_dataset','target_column','score']).to_csv(f'{R}/submission/sm_mapping.csv',index=False)
print(len(rows),'mappings')
