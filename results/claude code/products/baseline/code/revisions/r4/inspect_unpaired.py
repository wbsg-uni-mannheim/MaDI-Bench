import pandas as pd, sys
d=pd.read_pickle('state/s4_clustered.pkl')
sc=pd.read_pickle('state/s4_scored.pkl')
has=d.groupby('cluster_key').source.agg(set)
d['has2']=d.cluster_key.map(lambda k: 'dataset_2' in has[k]); d['has1']=d.cluster_key.map(lambda k: 'dataset_1' in has[k])
u1=d[(d.source=='dataset_1')&~d.has2]; u2=set(d[(d.source=='dataset_2')&~d.has1].index)
print(len(u1), len(u2), u1.product_type.value_counts().to_dict())
x=sc[(sc.src1=='dataset_1')&(sc.src2=='dataset_2')]
n=int(sys.argv[1]) if len(sys.argv)>1 else 40
for i,r in u1.head(n).iterrows():
    c=x[x.i==i].sort_values('score',ascending=False).head(2)
    print(f"* {r.n_title[:95]} [{r.n_storage_gb},{r.n_vram_gb},{r.n_mpn}]")
    for _,cc in c.iterrows():
        b=d.loc[cc.j]
        fs=' '.join(f'{k}={int(cc[k])}' for k in ['code','mpn','cap','chip','vram','brand','mem','mtok','rpm','iface','ffk','sasg','excl'] if cc[k]!=0)
        print(f"   {cc.score:.2f} {'FREE' if cc.j in u2 else 'taken'} {fs} | {b.n_title[:85]} [{b.n_storage_gb},{b.n_mpn}]")
