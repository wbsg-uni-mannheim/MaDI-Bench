"""Stage 1: schema matching. Writes submission/sm_mapping.csv and work/state/translated_<src>.csv"""
import pandas as pd, os
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
D = f"{ROOT}/task/input/data"
MAP = {
 'dbpedia': {'nm':'name','ly':'releaseYear','stu':'developer','sys':'platform','g':'genres','fran':'series'},
 'metacritic': {'game_title':'name','year_published':'releaseYear','made_by':'developer','console':'platform',
                'gnrs':'genres','press_rating':'criticScore','player_rating':'userScore','age_rating':'ESRB'},
 'sales': {'Attribute_2':'name','Attribute_3':'releaseYear','Attribute_4':'developer','Attribute_5':'publisher',
           'Attribute_6':'platform','Attribute_7':'genres','Attribute_8':'criticScore','Attribute_9':'userScore',
           'Attribute_10':'ESRB'},
}
# confidence: 1.0 unambiguous by examples; lower where semantics are inferred
CONF = {('metacritic','made_by'):0.85, ('sales','Attribute_4'):0.9, ('sales','Attribute_5'):0.9,
        ('dbpedia','stu'):0.85, ('dbpedia','fran'):0.9}
rows = []
for src, m in MAP.items():
    df = pd.read_csv(f"{D}/{src}.csv", dtype=str, keep_default_na=False)
    rows += [dict(source_dataset=src, source_column='id', target_dataset='target', target_column='id', score=1.0)]
    for c, t in m.items():
        rows.append(dict(source_dataset=src, source_column=c, target_dataset='target', target_column=t, score=CONF.get((src,c),1.0)))
    out = pd.DataFrame({'id': df['id']})
    for c, t in m.items():
        out[t] = df[c]
    out['source'] = src
    out.to_csv(f"{W}/state/translated_{src}.csv", index=False)
os.makedirs(f"{ROOT}/submission", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{ROOT}/submission/sm_mapping.csv", index=False)
print(pd.DataFrame(rows))
