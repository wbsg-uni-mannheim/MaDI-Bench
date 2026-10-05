"""Stage 6: attribute-wise fusion from final membership. Vote on comparison-normalized keys;
ties broken by per-attribute source priority chosen from the label-free odd-one-out diagnostic
(the source that most often disagrees with the other two in 3-source clusters ranks last)."""
import pandas as pd, numpy as np, os, re, json, unicodedata
from s2_normalize import fold, AFFIL
W=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(W); ST=f'{W}/state'
SRCS=['crossref','dblp','open_alex']
PRI={'type':['dblp','crossref','open_alex'],'title':['open_alex','crossref','dblp'],'authors':['crossref','dblp','open_alex'],
     'publication_year':['crossref','dblp','open_alex'],'journal':['open_alex','crossref','dblp'],
     'volume':['crossref','open_alex','dblp'],'issue':['crossref','open_alex','dblp'],'first_page':['crossref','open_alex','dblp'],
     'last_page':['crossref','open_alex','dblp'],'referenced_works_count':['open_alex','crossref','dblp'],'cited_by_count':['open_alex','crossref','dblp']}
DASH=str.maketrans({c:'-' for c in '‐‑‒–—―−'}); QUOTE=str.maketrans({'‘':"'",'’':"'",'“':'"','”':'"','`':"'"})
def tkey(t): return re.sub(r'\s+',' ',fold(t.translate(DASH).translate(QUOTE))).strip()
def name_key(n): return tuple(sorted(re.findall(r'[a-z0-9]+',fold(n.replace('.',' ').translate(DASH)))))
def name_match(a,b):
    a,b=set(a),set(b)
    return a==b or (len(a&b)>=2 and (a<=b or b<=a))
def list_sim(A,B):
    """fraction of names in the longer list matched by a name in the other (token-set equality or subset)"""
    ka=[name_key(n) for n in A]; kb=[name_key(n) for n in B]
    m=sum(1 for x in ka if any(name_match(x,y) for y in kb))
    return m/max(len(ka),len(kb))
def vote_authors(cands, pri):
    cands=[(s,v) for s,v in cands if isinstance(v,list) and v]
    if not cands: return None,None,0,0
    best=None
    for s,v in cands:
        sup=sum(list_sim(v,v2) for s2,v2 in cands)
        rank=(-round(sup,6), -sum(ord(ch)>127 for ch in ''.join(v)), pri.index(s))
        if best is None or rank<best[0]: best=(rank,v,s,sup)
    return best[1],best[2],round(best[3],3),len({akey(v) for _,v in cands})
def clean_auth(L):
    if not isinstance(L,list): return None
    L=[x for x in L if not AFFIL.search(x) and len(x)<=160]
    return L or None
def akey(L): return tuple(sorted(name_key(n) for n in L))
def jkey(j): return tuple(sorted(re.findall(r'[a-z0-9]+',fold(j))))
PAT=re.compile(r'^[A-Za-z0-9][A-Za-z0-9 ._/-]*$'); PPAT=re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
def vote(cands, pri, key=lambda v:v, support=None):
    """cands: list of (source,value). returns (value, source, n_support, n_distinct)"""
    cands=[(s,v) for s,v in cands if v is not None and not (isinstance(v,float) and np.isnan(v)) and v is not pd.NA]
    if not cands: return None,None,0,0
    keys=[key(v) for _,v in cands]
    best=None
    for (s,v),k in zip(cands,keys):
        sup=sum(1 for k2 in keys if (support(k2,k) if support else k2==k))
        rank=(-sup, pri.index(s))
        if best is None or rank<best[0]: best=(rank,v,s,sup)
    return best[1],best[2],best[3],len(set(keys))
def main():
    N=pd.concat([pd.read_pickle(f'{ST}/norm_{s}.pkl') for s in SRCS]).set_index('id')
    N['authors']=N.authors.map(clean_auth)
    M=pd.read_csv(f'{R}/submission/membership.csv')
    X=M.join(N.drop(columns='source'),on='record_id')
    out=[]; prov=[]
    for cid,g in X.groupby('cluster_id',sort=True):
        row={'_id':cid,'id':cid}; srcs=list(g.source)
        def c(col): return list(zip(srcs,g[col]))
        for att,col,kw in [('type','type',{}),('title','title',dict(key=tkey)),('authors','authors',{}),
                           ('publication_year','year',{}),('journal','journal',dict(key=jkey)),
                           ('volume','volume',{}),('issue','issue',{}),('first_page','first_page',{}),('last_page','last_page',{}),
                           ('referenced_works_count','referenced_works_count',{}),('cited_by_count','cited_by_count',{})]:
            v,s,sup,nd=(vote_authors if att=='authors' else vote)(c(col),PRI[att],**kw)
            if att=='journal' and s=='dblp':  # dblp abbreviation only when no full name exists (vote already prefers support)
                pass
            row[att]=v; prov.append((cid,att,s,sup,nd))
        out.append(row)
    F=pd.DataFrame(out)
    # schema validity: out-of-range -> null (never invented)
    F.loc[~F.publication_year.between(2018,2020),'publication_year']=pd.NA
    for col,p in [('volume',PAT),('issue',PAT),('first_page',PPAT),('last_page',PPAT)]:
        F[col]=F[col].map(lambda v: v if isinstance(v,str) and p.match(v) and len(v)<=32 else None)
    for col in ['first_page','last_page']:
        F[col]=F[col].map(lambda v: v if v is None or not v.isdigit() or 1<=int(v)<=100000 else None)
    bad=F.first_page.fillna('').str.isdigit()&F.last_page.fillna('').str.isdigit()
    bad&=pd.to_numeric(F.last_page,errors='coerce')<pd.to_numeric(F.first_page,errors='coerce')
    F.loc[bad,'last_page']=None; print('last<first nulled',bad.sum())
    F.loc[F.referenced_works_count>5000,'referenced_works_count']=pd.NA
    F.loc[F.cited_by_count>100000,'cited_by_count']=pd.NA
    F['authors']=F.authors.map(lambda L: json.dumps(L[:200],ensure_ascii=False) if isinstance(L,list) else None)
    F['title']=F.title.map(lambda t: t[:2000] if isinstance(t,str) else t)
    F['journal']=F.journal.map(lambda t: t[:300] if isinstance(t,str) else t)
    for col in ['publication_year','referenced_works_count','cited_by_count']: F[col]=F[col].astype('Int64')
    cols=['_id','id','type','title','authors','publication_year','journal','volume','issue','first_page','last_page','referenced_works_count','cited_by_count']
    F[cols].to_csv(f'{R}/submission/fused.csv',index=False)
    P=pd.DataFrame(prov,columns=['cluster_id','attribute','chosen_source','support','n_distinct'])
    P.to_csv(f'{ST}/fusion_provenance.csv',index=False)
    print(len(F),'fused rows'); print(F[cols].notna().mean().round(3).to_string())
    print('disagreement rate (n_distinct>1 among clusters with value):'); print(P[P.n_distinct>0].groupby('attribute').n_distinct.apply(lambda s:(s>1).mean()).round(3).to_string())
if __name__=='__main__': main()
