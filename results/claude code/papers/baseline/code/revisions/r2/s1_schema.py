"""Stage 1: schema matching. Map source-native columns to target attributes and
translate each source into a common (still raw-valued) frame."""
import json, pandas as pd, os
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
MAP = {
 'crossref': {'id':'id','work_type':'type','title_text':'title','contributor_names':'authors',
              'issued_year':'publication_year','container_title':'journal','volume_id':'volume',
              'issue_id':'issue','page_first':'first_page','page_last':'last_page',
              'reference_total':'referenced_works_count','cited_total':'cited_by_count'},
 'dblp': {'id':'id','entry_type':'type','publication_title':'title','author_list':'authors',
          'pub_year':'publication_year','venue_name':'journal','volume_no':'volume','issue_no':'issue',
          'page_start':'first_page','page_finish':'last_page'},
 'open_alex': {'id':'id','work_kind':'type','display_title':'title','authors_list':'authors',
               'year_published':'publication_year','source_name':'journal','volume_tag':'volume',
               'issue_tag':'issue','start_page':'first_page','end_page':'last_page',
               'refs_count':'referenced_works_count','citations_count':'cited_by_count'},
}
# unmapped (no target attribute): crossref publisher_name, abstract_text; open_alex topic_terms
SCORE = {('crossref','journal'):0.9,('dblp','journal'):0.8,('open_alex','journal'):0.9}
rows=[]
for s,m in MAP.items():
    for sc,tc in m.items():
        rows.append(dict(source_dataset=s,source_column=sc,target_dataset='target',target_column=tc,
                         score=SCORE.get((s,tc),1.0)))
os.makedirs('submission',exist_ok=True)
pd.DataFrame(rows).to_csv('submission/sm_mapping.csv',index=False)
tgt=json.load(open('task/input/schemamatching/target_schema.json'))['properties']
frames=[]
for s,m in MAP.items():
    d=pd.read_json(f'task/input/data/{s}.jsonl',lines=True,dtype=False)
    assert set(m)<=set(d.columns) and set(m.values())<=set(tgt)
    t=d[list(m)].rename(columns=m); t['source']=s
    frames.append(t)
T=pd.concat(frames,ignore_index=True)
T.to_pickle('work/state/s1_translated.pkl')
print(T.groupby('source').size())
