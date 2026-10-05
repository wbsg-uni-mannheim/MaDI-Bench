import pandas as pd, numpy as np, sys
sys.path.insert(0,'work')
from s3_blocking import topk_pairs
from s4_features import name_sim, artist_sim, track_sim
from sklearn.feature_extraction.text import TfidfVectorizer
src=sys.argv[1]
N=pd.read_pickle(f'work/state/norm_{src}.pkl').reset_index(drop=True)
na=(N.name_key+' | '+N.artist_key.fillna('')).str.strip()
X=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,3),sublinear_tf=True).fit_transform(na)
P=[(i,j,c) for i,j,c in topk_pairs(X,X,6,0.6) if i<j]
P=pd.DataFrame(P,columns=['i','j','cos'])
A=N.loc[P.i].reset_index(drop=True); B=N.loc[P.j].reset_index(drop=True)
P['ns']=[name_sim(a,b,x,y) for a,b,x,y in zip(A.name_key,B.name_key,A.name_dropmark,B.name_dropmark)]
P['as']=[artist_sim(a,b) for a,b in zip(A.artist_key,B.artist_key)]
t=[track_sim(a,b) for a,b in zip(A.track_core,B.track_core)]
P['ts']=[x[0] for x in t]; P['tc']=[x[1] for x in t]
P['dur_eq']=np.where(A.duration.notna()&B.duration.notna(), (A.duration-B.duration).abs()<=5, np.nan)
if src=='discogs':
    P['date_eq']=np.where(A.date.notna()&B.date.notna(), A.date==B.date, np.nan)
    P['ctry_eq']=np.where(A.country_key.notna()&B.country_key.notna(), A.country_key==B.country_key, np.nan)
    P['lab_eq']=[np.nan if not (isinstance(a,list) and isinstance(b,list)) else float(bool(set(map(str.lower,a))&set(map(str.lower,b)))) for a,b in zip(A.label,B.label)]
P['id1']=A.id; P['id2']=B.id
P.to_pickle(f'work/state/explore_within_{src}.pkl')
S=P[(P.ns>=0.9)&((P['as']>=0.7)|P['as'].isna())]
print(len(P), 'strong name+artist',len(S))
print('tracks cont dist',S.tc.describe().round(2).to_dict())
for c in ['dur_eq','date_eq','ctry_eq','lab_eq']:
    if c in S: print(c, S[c].value_counts(dropna=False).to_dict())
if src=='discogs':
    full=S[(S.tc>=0.9)]
    print('tc>=.9', len(full)); 
    for c in ['dur_eq','date_eq','ctry_eq','lab_eq']: print(' ',c, full[c].value_counts(dropna=False).to_dict())
    k=full[['date_eq','ctry_eq','lab_eq']].fillna(-1)
    print(pd.crosstab(k.date_eq,[k.ctry_eq,k.lab_eq]))
