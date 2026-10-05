import pandas as pd, numpy as np, sys
pd.set_option('display.width',250)
src=sys.argv[1]; lo=float(sys.argv[2]); hi=float(sys.argv[3]); n=int(sys.argv[4])
F=pd.read_pickle('work/state/scored.pkl'); M=pd.read_pickle('work/state/membership.pkl')
sz=M.groupby('cluster_id').size(); single=set(M[M.cluster_id.map(sz)==1].record_id)
N={s:pd.read_pickle(f'work/state/norm_{s}.pkl').set_index('id') for s in ['discogs','lastfm','musicbrainz']}
pref={'discogs':'discogs','lastFM':'lastfm','mbrainz':'musicbrainz'}
ids=[i for i in N[src].index if i in single]
G=F[F.id1.isin(ids)|F.id2.isin(ids)].copy()
G['me']=np.where(G.id1.isin(ids),G.id1,G.id2)
best=G.sort_values('score',ascending=False).groupby('me').head(1)
print('n unmatched',len(ids),'with cands',best.me.nunique())
print(pd.cut(best.score,[-20,0,2,3,4,4.5,6,20]).value_counts().sort_index().to_dict())
x=best[(best.score>=lo)&(best.score<hi)]
for r in x.sample(min(n,len(x)),random_state=2).itertuples():
    a=N[pref[r.id1.split('_')[0]]].loc[r.id1]; b=N[pref[r.id2.split('_')[0]]].loc[r.id2]
    ca=M.set_index('record_id').cluster_id[r.id2 if r.me==r.id1 else r.id1]
    print(f"{r.score:4.1f} n{r.p_name} a{r.p_artist} t{r.p_tracks} d{r.p_dur} dt{r.p_date} c{r.p_ctry} | {a['name']} / {a.artist} [{a.n_tracks},{a.duration},{a.date},{a.country}] <> {b['name']} / {b.artist} [{b.n_tracks},{b.duration},{b.date},{b.country}] other_cl={sz[ca]}")
