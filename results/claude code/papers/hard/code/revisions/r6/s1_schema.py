"""Stage 1: schema matching. Translate each source into target attribute names (raw values kept)."""
import pandas as pd, json, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = {
 'crossref': {'typ':'type','ttl':'title','auth_list':'authors','pub_year':'publication_year','venue':'journal',
              'vol':'volume','iss_no':'issue','pg_first':'first_page','pg_last':'last_page',
              'ref_count':'referenced_works_count','cited_count':'cited_by_count'},
 'dblp': {'ty':'type','t':'title','au':'authors','py':'publication_year','j':'journal','v':'volume','i':'issue',
          'fp':'first_page','lp':'last_page','rc':'referenced_works_count','cc':'cited_by_count'},
 'open_alex': {'Attribute_3':'type','Attribute_4':'title','Attribute_5':'authors','Attribute_6':'publication_year',
               'Attribute_7':'journal','Attribute_10':'volume','Attribute_11':'issue','Attribute_12':'first_page',
               'Attribute_13':'last_page','Attribute_14':'referenced_works_count','Attribute_15':'cited_by_count'},
}
UNMAPPED = {'crossref':{'publ':'publisher','abstract_text':'abstract','kw_list':'keywords'},
            'dblp':{'pb':'publisher','kw':'keywords'},
            'open_alex':{'Attribute_8':'publisher','Attribute_9':'keywords'}}
TARGET = ['type','title','authors','publication_year','journal','volume','issue','first_page','last_page',
          'referenced_works_count','cited_by_count']
if __name__ == '__main__':
    schema = json.load(open(f'{BASE}/task/input/schemamatching/target_schema.json'))
    assert set(TARGET) <= set(schema['properties'])
    rows = []
    frames = []
    for src, m in MAP.items():
        d = pd.read_csv(f'{BASE}/task/input/data/{src}.csv', dtype=str, keep_default_na=False)
        assert set(m) <= set(d.columns)
        out = pd.DataFrame({'id': d['id'], 'source': src})
        for sc, tc in m.items():
            out[tc] = d[sc]
        for sc, nm in UNMAPPED[src].items():
            out['x_' + nm] = d[sc]
        frames.append(out)
        rows.append((src, 'id', 'papers', 'id', 1.0))
        for sc, tc in m.items():
            rows.append((src, sc, 'papers', tc, 1.0))
    u = pd.concat(frames, ignore_index=True)
    u.to_pickle(f"{BASE}/work/state/s1_translated.pkl")
    os.makedirs(f'{BASE}/submission', exist_ok=True)
    pd.DataFrame(rows, columns=['source_dataset','source_column','target_dataset','target_column','score']).to_csv(
        f'{BASE}/submission/sm_mapping.csv', index=False)
    print(u.shape, u.source.value_counts().to_dict())
