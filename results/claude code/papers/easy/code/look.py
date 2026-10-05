import pandas as pd, numpy as np, sys
C=pd.read_pickle('state/scored.pkl')
D=pd.concat([pd.read_pickle(f'state/norm_{s}.pkl') for s in ['crossref','dblp','open_alex']]).set_index('id')
lo,hi,n=float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3])
# only pairs that are best for id1 among its candidates in that source
C['r1']=C.groupby(['id1','s2']).score.rank(ascending=False,method='first')
C['r2']=C.groupby(['id2','s1']).score.rank(ascending=False,method='first')
x=C[(C.score>=lo)&(C.score<hi)&(C.r1==1)&(C.r2==1)]
print('mutual-best in band',len(x))
for _,r in x.sample(min(n,len(x)),random_state=int(sys.argv[4]) if len(sys.argv)>4 else 0).iterrows():
    print(f"--- {r.score:.2f} T={r['T']:.2f} aov={r.aov} yd={r.ydiff} jc={r.jc} vol={r.vol} fp={r.fp} typ={r.typ}")
    for i in [r.id1,r.id2]:
        d=D.loc[i]; print('  ',i,d.type,d.year,'|',str(d.title)[:90],'|',str(d.authors)[:60],'|',d.journal,d.volume,d.first_page)
