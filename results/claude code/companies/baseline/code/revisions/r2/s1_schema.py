"""Stage 1: schema matching. Semantic mapping chosen by inspecting names, metadata, types and example values."""
import pandas as pd, json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
T = 'companies'
M = [
 # source, column, target, score, transformation / reason
 ('dbpedia','entity_uri','id',1.0,'native id (DBpedia URI)'),
 ('dbpedia','org_name','name',1.0,'strip legal suffix for fused name'),
 ('dbpedia','established','founded',1.0,'ISO date; year-level (all -01-01)'),
 ('dbpedia','nation','country',1.0,'map to CLDR country name'),
 ('dbpedia','headquarters','city',0.8,'concatenated location string; city extracted heuristically'),
 ('dbpedia','sector','industry',0.8,'free-text sector keyword-mapped to GICS industry'),
 ('dbpedia','keypeople_name','keypeople',0.9,'metadata: founders; python-list string parsed'),
 ('dbpedia','total_assets_val','assets',0.8,'monetary amount, unit not stated'),
 ('dbpedia','annual_income','revenue',0.7,'metadata says income; values revenue-scale; unit not stated'),
 ('forbes','forbes_url','id',1.0,'native id'),
 ('forbes','company','name',1.0,''),
 ('forbes','region','country',1.0,'ISO-style long names mapped to CLDR'),
 ('forbes','business_segment','industry',0.9,'Forbes industry mapped to GICS industry by manual table'),
 ('forbes','asset_value','assets',1.0,'already US$ units'),
 ('forbes','sales_figure','revenue',1.0,'sales = revenue, US$'),
 ('fullcontact','Attribute_1','id',1.0,'values fullcontact_N'),
 ('fullcontact','Attribute_2','name',1.0,'company names'),
 ('fullcontact','Attribute_3','country',1.0,'country names'),
 ('fullcontact','Attribute_4','city',1.0,'city names'),
 ('fullcontact','Attribute_5','keypeople',0.9,'person names (founders/key persons), list strings'),
 ('fullcontact','Attribute_6','founded',1.0,'ISO dates -01-01'),
]
df = pd.DataFrame(M, columns=['source_dataset','source_column','target_column','score','note'])
df['target_dataset'] = T
schema = json.load(open('task/input/schemamatching/target_schema.json'))
for s in ['dbpedia','forbes','fullcontact']:
    cols = pd.read_csv(f'task/input/data/{s}.csv', nrows=1).columns
    used = set(df[df.source_dataset==s].source_column)
    assert used <= set(cols), (s, used-set(cols))
    print(s, 'unmapped:', set(cols)-used)
assert set(df.target_column) <= set(schema['properties'])
print('targets without contributor:', set(schema['properties'])-set(df.target_column))
df.to_csv('work/state/sm_inventory.csv', index=False)
df[['source_dataset','source_column','target_dataset','target_column','score']].to_csv('submission/sm_mapping.csv', index=False)
