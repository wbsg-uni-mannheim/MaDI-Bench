"""Stage 1: schema matching. Writes sm_mapping.csv and a unified raw table (values untouched)."""
import pandas as pd, json, os
BASE = os.path.dirname(os.path.abspath(__file__)) + '/..'
DATA = f'{BASE}/task/input/data'
# source column -> target attribute (decided by inspecting meanings/examples, see report)
MAP = {
 'dbpedia': {'gm_nm':'name','launch_yr':'releaseYear','studio':'developer','system':'platform','gnr':'genres','franchise':'series'},
 'metacritic': {'name':'name','releaseYear':'releaseYear','developer':'developer','platform':'platform','genres':'genres',
                'criticScore':'criticScore','userScore':'userScore','ESRB':'ESRB'},
 'sales': {'prod_title':'name','launch_dt':'releaseYear','studio':'developer','dist':'publisher','hw':'platform','genre':'genres',
           'press_score':'criticScore','comm_rating':'userScore','age_class':'ESRB'},
}
# confidence: 1.0 unless semantics slightly differ
SCORE = {('dbpedia','launch_yr'):0.95, ('dbpedia','franchise'):0.9, ('sales','dist'):0.95, ('sales','studio'):0.95,
         ('sales','press_score'):0.95, ('sales','comm_rating'):0.95, ('sales','age_class'):0.95}
TARGET = ['name','releaseYear','developer','genres','publisher','platform','criticScore','userScore','ESRB','series']
rows=[]; frames=[]
for src, m in MAP.items():
    df = pd.read_csv(f'{DATA}/{src}.csv', dtype=str, keep_default_na=False)
    for c,t in m.items():
        assert c in df.columns and t in TARGET
        rows.append((src,c,'videogames',t,SCORE.get((src,c),1.0)))
    rows.append((src,'id','videogames','id',1.0))
    u = pd.DataFrame({'rid':df['id'],'source':src})
    for t in TARGET:
        inv = [c for c,tt in m.items() if tt==t]
        u[t] = df[inv[0]].replace('', None) if inv else None
    frames.append(u)
os.makedirs(f'{BASE}/submission', exist_ok=True)
pd.DataFrame(rows, columns=['source_dataset','source_column','target_dataset','target_column','score']).to_csv(f'{BASE}/submission/sm_mapping.csv', index=False)
U = pd.concat(frames, ignore_index=True)
assert U.rid.is_unique
U.to_pickle(f'{BASE}/work/state/s1_unified.pkl')
print(U.groupby('source').count())
