import pandas as pd, sys
d=pd.read_pickle('state/s4_clustered.pkl').set_index('id')
C=pd.read_csv('../submission/correspondences.csv',dtype={'id1':str,'id2':str})
sc=pd.read_pickle('state/s4_scored.pkl').set_index(['id1','id2'])
f=sys.argv[1]; n=int(sys.argv[2])
k=0
for a,b in zip(C.id1,C.id2):
    key=(a,b) if (a,b) in sc.index else ((b,a) if (b,a) in sc.index else None)
    if key is None: continue
    r=sc.loc[key]
    if (f=='excl' and r[f]>0) or (f!='excl' and r[f]<0):
        A=d.loc[a];B=d.loc[b]
        print(f"{r.score:.2f} | {A.n_title[:90]} [{A.n_mpn}|{A.rpm}|{A.n_mem}]\n     | {B.n_title[:90]} [{B.n_mpn}|{B.rpm}|{B.n_mem}]")
        k+=1
        if k>=n: break
