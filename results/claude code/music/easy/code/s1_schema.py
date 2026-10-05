import pandas as pd, json
IN='task/input/data/'
schema=json.load(open('task/input/schemamatching/target_schema.json'))
targets=list(schema['properties'])
rows=[]
for s in ['discogs','lastfm','musicbrainz']:
    cols=pd.read_csv(IN+s+'.csv',nrows=5).columns
    for c in cols:
        # identical semantics verified by inspecting values: names/types match target attributes
        assert c in targets, c
        rows.append(dict(source_dataset=s,source_column=c,target_dataset='target',target_column=c,score=1.0))
pd.DataFrame(rows).to_csv('submission/sm_mapping.csv',index=False)
print(pd.DataFrame(rows).groupby('source_dataset').target_column.apply(list))
