"""Stage 4b: interpretable additive evidence score (no training). See report for rationale."""
import pandas as pd, numpy as np, os
W=os.path.dirname(os.path.abspath(__file__)); ST=f'{W}/state'
def score(C):
    """T: best title similarity (char ratio, token-set, space-insensitive ratio, prefix = dropped subtitle).
    E: small bounded adjustments from independent fields; missing = unknown (0)."""
    T=np.maximum.reduce([C.tr.fillna(0).values, 0.95*C.tset.fillna(0).values, C.trn.fillna(0).values])
    T=np.where((C.prefix==1)|(C.pfxn==1), np.maximum(T,0.85), T)
    E=np.zeros(len(C))
    a=C.aov
    E+=np.where(a.isna(),0,np.where(a==0,-0.3,0.1*(a-0.5)))
    y=C.ydiff
    E+=np.where(y.isna(),0,np.where(y==0,0.03,np.where(y==1,-0.02,-0.08)))
    E+=np.where(C.jc.isna(),0,np.where(C.jc>=0.9,0.05,-0.05))
    hard_eq=np.zeros(len(C)); hard_neq=np.zeros(len(C))
    for f in ['vol','fp','lp','iss']:
        hard_eq+=(C[f]==1).values; hard_neq+=(C[f]==0).values
    E+=0.05*np.minimum(hard_eq,2)-0.1*hard_neq
    cd=(C.s1=='crossref')&(C.s2=='dblp')
    E+=np.where(cd&C.typ.notna(),np.where(C.typ==1,0.03,-0.1),0)
    E+=np.where(C.refd.isna(),0,np.where(C.refd<0.2,0.03,np.where(C.refd>0.5,-0.03,0)))
    S=T+E
    # generic one-word (possibly truncated) titles: title evidence is weak, require corroboration
    # ...or unique prefix match within same venue+year (truncated title, e.g. 'Extended' in IET Biom. 2018)
    pv=(C.generic_short&((C.prefix==1)|(C.pfxn==1))&(C.jc>=0.9)&(C.ydiff==0)&~(a<0.5)).values
    uniq=np.zeros(len(C),bool)
    if pv.any():
        P=C.loc[pv,['id1','id2','s1','s2']]
        n1=P.groupby(['id1','s2']).id2.transform('size'); n2=P.groupby(['id2','s1']).id1.transform('size')
        uniq[np.where(pv)[0]]=((n1==1)&(n2==1)).values
    weak=C.generic_short.values&~((a.fillna(0)>=0.5).values|(a.isna().values&(hard_eq>=2))|uniq)
    S=np.where(weak,np.minimum(S,0.8),S)
    # title neither near-equal nor prefix: accept only token-subset titles with author support, or >=2 hard ids
    tw=((C.trn.fillna(0)<0.8)&(C.prefix==0)&(C.pfxn==0)).values
    tw_ok=((C.tset.fillna(0)>=0.95)&(a.fillna(0)>=0.5)).values|(hard_eq>=2)
    S=np.where(tw&~tw_ok,np.minimum(S,0.8),S)
    ACC=(S>=THR)|((T>=0.6)&(hard_eq>=2)&(hard_neq==0)&(a.fillna(0)>=0.5).values)
    return S, T, ACC
THR=0.88
if __name__=='__main__':
    C=pd.read_pickle(f'{ST}/features.pkl')
    import re
    N=pd.concat([pd.read_pickle(f'{ST}/norm_{s}.pkl') for s in ['crossref','dblp','open_alex']]).set_index('id')
    def generic(t):
        if not isinstance(t,str): return True
        w=t.split()
        return len(w)<=1 and not re.search(r'[A-Z].*[A-Z]|\d|[a-z][A-Z]',t)
    g=N.title.map(generic).to_dict()
    C['generic_short']=[g[x] or g[y] for x,y in zip(C.id1,C.id2)]
    print('pairs with generic short title',C.generic_short.sum())
    C['score'],C['T'],C['acc']=score(C)
    print('accepted',C.acc.sum())
    C.to_pickle(f'{ST}/scored.pkl')
    print(np.histogram(C.score,bins=np.arange(0,1.5,0.05)))
