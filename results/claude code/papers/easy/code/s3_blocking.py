"""Stage 3: blocking. Union of (a) exact normalized-title key, (b) char-3gram TF-IDF top-K
cosine neighbours in both directions per source pair, (c) author-token+year key for
records with short/missing titles."""
import pandas as pd, numpy as np, os, itertools, time
from s2_normalize import fold
from sklearn.feature_extraction.text import TfidfVectorizer
W=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(W); ST=f'{W}/state'
SRCS=['crossref','dblp','open_alex']; K=10; MINCOS=0.3
D={s:pd.read_pickle(f'{ST}/norm_{s}.pkl') for s in SRCS}
allk=pd.concat([D[s].tkey for s in SRCS])
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,3),min_df=2,sublinear_tf=True,dtype=np.float32).fit(allk)
X={s:vec.transform(D[s].tkey) for s in SRCS}
def topk(A,B,k):
    out=[]
    BT=B.T.tocsc()
    for i in range(0,A.shape[0],2000):
        S=(A[i:i+2000]@BT).toarray()
        idx=np.argpartition(-S,k,axis=1)[:,:k]
        val=np.take_along_axis(S,idx,1)
        r,c=np.nonzero(val>=MINCOS)
        out.append(np.stack([r+i,idx[r,c],val[r,c]],1))
    return np.concatenate(out)
pairs=[]
for a,b in itertools.combinations(SRCS,2):
    t=time.time(); A,B=D[a],D[b]
    P=set()
    src_cnt={}
    # (a) exact title key
    m=A[['id','tkey']].merge(B[['id','tkey']],on='tkey'); m=m[m.tkey.str.len()>0]
    ex=set(zip(m.id_x,m.id_y)); P|=ex
    # (b) tfidf both directions
    t1=topk(X[a],X[b],K); t2=topk(X[b],X[a],K)
    tf=set(zip(A.id.values[t1[:,0].astype(int)],B.id.values[t1[:,1].astype(int)]))|set(zip(A.id.values[t2[:,1].astype(int)],B.id.values[t2[:,0].astype(int)]))
    P|=tf
    # (c) short/missing titles (subtitle dropped, truncation): first-title-token + year with prefix relation,
    #     and pairs of author tokens + year
    def short(df): return df[df.tkey.str.split().str.len().fillna(0)<4]
    def ft(df):
        x=df[df.tkey.str.len()>0][['id','tkey','year']].copy(); x['k']=x.tkey.str.split().str[0]+'|'+x.year.astype(str); return x
    fa,fb=ft(A),ft(B); sa,sb=ft(short(A)),ft(short(B))
    m=pd.concat([sa.merge(fb,on='k'),fa.merge(sb,on='k')])
    m=m[[x.startswith(y) or y.startswith(x) for x,y in zip(m.tkey_x,m.tkey_y)]]
    pre=set(zip(m.id_x,m.id_y))
    def ap(df):
        rows=[]
        for i,L,y in zip(df.id,df.authors,df.year):
            if not L or pd.isna(y): continue
            toks=sorted({w for x in L[:6] for w in __import__('re').findall(r'[a-z]+',fold(x)) if len(w)>2})
            rows+=[(i,f'{p}|{q}|{y}') for p,q in itertools.combinations(toks,2)]
        return pd.DataFrame(rows,columns=['id','k'])
    pa,pb=ap(A),ap(B); spa=pa[pa.id.isin(set(short(A).id))]; spb=pb[pb.id.isin(set(short(B).id))]
    m=pd.concat([spa.merge(pb,on='k'),pa.merge(spb,on='k')])
    au=set(zip(m.id_x,m.id_y))
    P|=pre|au
    print(a,b,'exact',len(ex),'tfidf',len(tf),'prefix',len(pre),'author-pairs',len(au),'union',len(P),'reduction',1-len(P)/(len(A)*len(B)),f'{time.time()-t:.0f}s')
    pairs+= [(x,y,a,b) for x,y in P]
C=pd.DataFrame(pairs,columns=['id1','id2','s1','s2'])
C.to_pickle(f'{ST}/candidates.pkl')
os.makedirs(f'{R}/submission/blocking',exist_ok=True)
C[['id1','id2']].to_csv(f'{R}/submission/blocking/candidates.csv',index=False)
# coverage diagnostics
for s in SRCS:
    ids=set(C.id1)|set(C.id2); print(s,'records with >=1 candidate',D[s].id.isin(ids).mean().round(4))
