import pandas as pd, sys
d=pd.read_pickle('state/s4_clustered.pkl')
L=pd.read_csv('state/s4_assign_log.csv')
c=pd.read_pickle('state/s4_scored.pkl').set_index(['i','j'])
step=sys.argv[1]; lo=float(sys.argv[2]); hi=float(sys.argv[3]); n=int(sys.argv[4])
x=L[(L.step==step)&(L.score>lo)&(L.score<hi)]
x=x.sample(min(n,len(x)),random_state=0).sort_values('score')
for _,r in x.iterrows():
    a=d.loc[r.record]; b=d.loc[r.cluster]
    key=(min(r.record,r.cluster),max(r.record,r.cluster))
    f=c.loc[key] if key in c.index else None
    fs='' if f is None else ' '.join(f'{k}={int(f[k])}' for k in ['code','mpn','cap','chip','vram','brand','mem','mtok','rpm','iface','ffk','line_diff'] if f[k]!=0)
    print(f"{r.score:.2f} {r.decision[:6]} {fs}\n   {a.n_title[:100]} [{a.n_storage_gb},{a.n_vram_gb},{a.n_chip},{a.n_mpn}]\n   {b.n_title[:100]} [{b.n_storage_gb},{b.n_vram_gb},{b.n_chip},{b.n_mpn}]")
