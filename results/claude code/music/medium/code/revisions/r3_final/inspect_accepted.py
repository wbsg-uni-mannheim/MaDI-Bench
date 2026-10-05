import pandas as pd, numpy as np, sys
q=sys.argv[1]; n=int(sys.argv[2])
F=pd.read_pickle('work/state/scored.pkl'); A=pd.read_pickle('work/state/accepted_edges.pkl')
X=F.merge(A[['id1','id2']],on=['id1','id2'])
N={s:pd.read_pickle(f'work/state/norm_{s}.pkl').set_index('id') for s in ['discogs','lastfm','musicbrainz']}
pref={'discogs':'discogs','lastFM':'lastfm','mbrainz':'musicbrainz'}
x=X.query(q); print('n',len(x))
for r in x.sample(min(n,len(x)),random_state=3).itertuples():
    a=N[pref[r.id1.split('_')[0]]].loc[r.id1]; b=N[pref[r.id2.split('_')[0]]].loc[r.id2]
    print(f"{r.score:4.1f} n{r.p_name} a{r.p_artist} t{r.p_tracks} d{r.p_dur} dt{r.p_date} c{r.p_ctry} | {a['name']} / {a.artist} [{a.n_tracks},{a.duration},{a.date},{a.country}] <> {b['name']} / {b.artist} [{b.n_tracks},{b.duration},{b.date},{b.country}]")
