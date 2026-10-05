"""Stage 4a: pairwise evidence for every blocking candidate."""
import pandas as pd, numpy as np, os, re, time
from rapidfuzz import fuzz
from s2_normalize import fold
W=os.path.dirname(os.path.abspath(__file__)); ST=f'{W}/state'
SRCS=['crossref','dblp','open_alex']
D=pd.concat([pd.read_pickle(f'{ST}/norm_{s}.pkl') for s in SRCS]).set_index('id')
STOP={'of','and','the','on','in','for','a','an','de','und','la','&','to','with'}
def jtoks(j): return [w for w in re.findall(r'[a-z0-9]+',fold(j)) if w not in STOP]
def jcompat(a,b):
    """1 if journal names compatible (equal tokens, or one is an abbreviation of the other in order), 0 otherwise"""
    ta,tb=jtoks(a),jtoks(b)
    if not ta or not tb: return np.nan
    if ta==tb: return 1.0
    s,l=(ta,tb) if len(''.join(ta))<=len(''.join(tb)) else (tb,ta)
    i=0
    for w in l:
        if i<len(s) and w.startswith(s[i]): i+=1
    if i==len(s): return 1.0
    return fuzz.token_set_ratio(' '.join(ta),' '.join(tb))/100 if fuzz.token_set_ratio(' '.join(ta),' '.join(tb))>=90 else 0.0
def eq(a,b):
    if a is None or b is None or (isinstance(a,float) and np.isnan(a)) or (isinstance(b,float) and np.isnan(b)) or a is pd.NA or b is pd.NA: return np.nan
    return float(str(a).lower()==str(b).lower())
C=pd.read_pickle(f'{ST}/candidates.pkl')
t=time.time()
cols={k:D[k].to_dict() for k in ['tkey','akeys','year','journal','volume','issue','first_page','last_page','type','referenced_works_count','cited_by_count']}
F={k:[] for k in ['trn','pfxn','tr','tset','prefix','aov','ydiff','jc','vol','iss','fp','lp','typ','refd']}
for x,y in zip(C.id1.values,C.id2.values):
    a,b=cols['tkey'][x],cols['tkey'][y]
    F['tr'].append(fuzz.ratio(a,b)/100 if a and b else np.nan)
    F['tset'].append(fuzz.token_set_ratio(a,b)/100 if a and b else np.nan)
    na,nb=a.replace(' ',''),b.replace(' ','')
    F['trn'].append(fuzz.ratio(na,nb)/100 if na and nb else np.nan)
    F['pfxn'].append(float(len(na)>=4 and len(nb)>=4 and na!=nb and (na.startswith(nb) or nb.startswith(na))))
    F['prefix'].append(float(bool(a) and bool(b) and a!=b and (a.startswith(b) or b.startswith(a))))
    ka,kb=cols['akeys'][x],cols['akeys'][y]
    F['aov'].append(len(ka&kb)/min(len(ka),len(kb)) if ka and kb else np.nan)
    ya,yb=cols['year'][x],cols['year'][y]
    F['ydiff'].append(abs(int(ya)-int(yb)) if not(pd.isna(ya) or pd.isna(yb)) else np.nan)
    ja,jb=cols['journal'][x],cols['journal'][y]
    F['jc'].append(jcompat(ja,jb) if ja and jb else np.nan)
    for f,k in [('vol','volume'),('iss','issue'),('fp','first_page'),('lp','last_page'),('typ','type')]:
        F[f].append(eq(cols[k][x],cols[k][y]))
    ra,rb=cols['referenced_works_count'][x],cols['referenced_works_count'][y]
    F['refd'].append(abs(int(ra)-int(rb))/max(int(ra),int(rb),5) if not(pd.isna(ra) or pd.isna(rb)) else np.nan)
for k,v in F.items(): C[k]=np.array(v,dtype=np.float32)
C.to_pickle(f'{ST}/features.pkl'); print(len(C),f'{time.time()-t:.0f}s')
print(C.describe().T.to_string())
