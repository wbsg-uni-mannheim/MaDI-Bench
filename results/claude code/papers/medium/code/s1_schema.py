"""Stage 1: schema mapping. Rename source columns to target names; keep raw values."""
import pandas as pd, json
D = 'task/input/data/'
MAP = {
 'crossref': {'id':'id','type':'type','title':'title','authors':'authors','publication_year':'publication_year',
   'journal':'journal','publisher':'publisher','abstract_text':'abstract_text','volume':'volume','issue':'issue',
   'first_page':'first_page','last_page':'last_page','referenced_works_count':'referenced_works_count',
   'cited_by_count':'cited_by_count','keywords':'keywords'},
 'dblp': {'id':'id','typ':'type','ttl':'title','auth':'authors','pub_yr':'publication_year','jrnl':'journal',
   'vol':'volume','iss':'issue','fpage':'first_page','lpage':'last_page','pub':'publisher','kwds':'keywords',
   'ref_cnt':'referenced_works_count','cite_cnt':'cited_by_count'},
 'open_alex': {'id':'id','wt':'type','tt':'title','aus':'authors','pyr':'publication_year','jrn':'journal',
   'kwd':'keywords','vol':'volume','isu':'issue','fpa':'first_page','lpa':'last_page','rwn':'referenced_works_count',
   'cby':'cited_by_count','pbl':'publisher'},
}
TARGET = set(json.load(open('task/input/schemamatching/target_schema.json'))['properties'])
frames, rows = [], []
for s, m in MAP.items():
    df = pd.read_csv(D + s + '.csv', dtype=str, keep_default_na=False)
    assert set(m) == set(df.columns), (s, set(df.columns) ^ set(m))
    df = df.rename(columns=m); df['source'] = s
    frames.append(df)
    for sc, tc in m.items():
        if tc in TARGET:
            rows.append((s, sc, 'papers', tc, 1.0))
u = pd.concat(frames, ignore_index=True).fillna('')
u.to_pickle('work/state/s1_unified_raw.pkl')
pd.DataFrame(rows, columns=['source_dataset','source_column','target_dataset','target_column','score']).to_csv('submission/sm_mapping.csv', index=False)
print(u.shape, u.source.value_counts().to_dict())
