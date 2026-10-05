"""Stage 4: pairwise entity matching with interpretable rules (no training, no labels).
name class (N1..N5) x evidence score e from non-name attributes; thresholds documented in report."""
import pandas as pd, numpy as np, re, json
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
U=pd.read_pickle('work/state/s2_normalized.pkl')
C=pd.read_pickle('work/state/s3_candidates.pkl')
R=U.set_index('record_id')
a=R.loc[C.id1].reset_index(); b=R.loc[C.id2].reset_index()
# harmless extra tokens for containment matches: descriptive words that do not by themselves denote a different entity
HARMLESS={'group','holding','holdings','international','intl','industries','inds','industrial','technologies','technology','tech',
 'solutions','products','foods','enterprises','corporation','company','the','and','of','co','incorporated','limited','plc','inc','corp','ltd','worldwide','global','brands'}
def toks(s): return set(s.split())
def people_key(pl):
    if not isinstance(pl,list): return None
    out=set()
    for p in pl:
        w=re.sub(r'[^a-z ]','',p.lower()).split()
        if w: out.add(w[-1])
    return out or None
def ev(r1,r2,s1,s2):
    e=0.0; notes=[]
    fc='fullcontact' in (s1,s2)
    c1,c2=r1.country,r2.country
    if isinstance(c1,str) and isinstance(c2,str):
        if c1==c2: e+=1; notes.append('ctry+')
        elif {c1,c2}<={'China','Hong Kong SAR China'} or {c1,c2}<={'China','Taiwan'}: pass
        else: e-= 0.3 if fc else 0.7; notes.append('ctry-')
    if isinstance(r1.paren_country,str) or isinstance(r2.paren_country,str):
        pc=r1.paren_country if isinstance(r1.paren_country,str) else r2.paren_country
        other=r2.country if isinstance(r1.paren_country,str) else r1.country
        if isinstance(other,str) and other!=pc: e-=1; notes.append('paren-')
    t1,t2=r1.city,r2.city
    if isinstance(t1,str) and isinstance(t2,str):
        x,y=set(t1.lower().split()),set(t2.lower().split())
        if x<=y or y<=x: e+=1; notes.append('city+')
        else: e-=0.3; notes.append('city-')
    y1,y2=r1.founded_year,r2.founded_year
    if pd.notna(y1) and pd.notna(y2):
        if abs(y1-y2)<=1: e+=1; notes.append('yr+')
        else: e-=1.5; notes.append('yr-')
    p1,p2=people_key(r1.keypeople),people_key(r2.keypeople)
    if p1 and p2:
        if p1&p2: e+=1.5; notes.append('kp+')
        else: e-=1; notes.append('kp-')
    for att in ['revenue','assets']:
        v1,v2=r1[att],r2[att]
        if pd.notna(v1) and pd.notna(v2) and v1>0 and v2>0:
            q=max(v1,v2)/min(v1,v2)
            if q<=1.25: e+=1; notes.append(att[:3]+'+')
            elif q>3: e-=1; notes.append(att[:3]+'-')
    if isinstance(r1.paren_desc,str)!=isinstance(r2.paren_desc,str): e-=0.5; notes.append('pdesc-')
    if isinstance(r1.industry,str) and isinstance(r2.industry,str) and r1.industry==r2.industry: e+=0.5; notes.append('ind+')
    return e,notes
rows=[]
for i in range(len(C)):
    r1,r2=a.iloc[i],b.iloc[i]
    s1,s2=r1.source,r2.source
    k1,k2=r1.k,r2.k; l1,l2=r1.k_loose,r2.k_loose
    T1,T2=toks(k1),toks(k2)
    cls=None; ns=0
    if (T1==T2 or k1.replace(' ','')==k2.replace(' ','')) and k1: cls,ns='N1',10
    elif toks(l1)==toks(l2) and l1: cls,ns='N2',8
    else:
        j1,j2=k1.replace(' ',''),l2 and k2.replace(' ','')
        r=fuzz.ratio(k1.replace(' ',''),k2.replace(' ',''))
        sub=(T1<T2 or T2<T1)
        extra=(T1^T2)
        lev=Levenshtein.distance(k1.replace(' ',''),k2.replace(' ',''))
        if r>=90 and min(len(j1),len(j2))>=8 and k1[0]==k2[0] and lev<=2: cls,ns=('N3a' if lev==1 else 'N3b'),6
        elif sub and extra<=HARMLESS and len(T1&T2-HARMLESS)>=1 and min(len(k1),len(k2))>=4: cls,ns='N4',6
        elif 'acronym' in C.how.iat[i]: cls,ns='N5',4
    if cls is None: continue
    e,notes=ev(r1,r2,s1,s2)
    thr={'N1':-1.5,'N2':-0.3,'N3a':0.5,'N3b':1.5,'N4':1.0,'N5':1.0}[cls]
    acc=e>=thr
    if 'pdesc-' in notes and not any(n.endswith('+') for n in notes): acc=False
    if cls=='N5' and not ('ctry+' in notes and r1.country!='United States'): acc=False
    rows.append(dict(id1=C.id1.iat[i],id2=C.id2.iat[i],s1=s1,s2=s2,n1=r1.name_raw,n2=r2.name_raw,cls=cls,e=e,notes=' '.join(notes),
                     accept=acc,score=ns+e))
M=pd.DataFrame(rows)
M.to_pickle('work/state/s4_scored.pkl')
print(M.groupby(['cls','accept']).size())
print(M[M.accept].groupby(['s1','s2']).size())
