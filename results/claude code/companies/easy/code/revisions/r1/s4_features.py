"""Stage 4a: pairwise evidence for every candidate."""
import os, numpy as np, pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
W=os.path.dirname(os.path.abspath(__file__))
N=pd.read_csv(f"{W}/state/normalized.csv"); C=pd.read_csv(f"{W}/state/candidates.csv")
E=np.load(f"{W}/state/emb_name.npy"); pos={r:i for i,r in enumerate(N.record_id)}
N["keys"]=N.apply(lambda r:({r.name_key} if isinstance(r.name_key,str) else set())|(set(r.alt_key.split("|")) if isinstance(r.alt_key,str) else set()),axis=1)
# token idf over name_key tokens for weighting
from collections import Counter
df=Counter(t for k in N.name_key.fillna("") for t in set(k.split())); ND=len(N)
idf=lambda t: np.log(ND/(1+df.get(t,0)))
GEO=set("""america americas american usa us north south east west asia asian pacific europe european africa india indian canada canadian
uk china chinese japan japanese australia australian germany deutschland france brasil brazil mexico singapore hong kong korea taiwan
ireland italia italy espana spain nederland netherlands international global worldwide middle emirates uae philippines malaysia
thailand indonesia vietnam russia poland turkey israel sweden norway denmark finland switzerland austria belgium new zealand nz
argentina chile colombia peru africa nigeria kenya egypt saudi arabia qatar pakistan bangladesh""".split())
out=[]
GREATER_CHINA={'China','Hong Kong SAR China','Macao SAR China'}
import unicodedata
def fold(s):
    s=unicodedata.normalize('NFKD',s); return ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()
for r in C.itertuples():
    a,b=N.iloc[pos[r.id1]],N.iloc[pos[r.id2]]
    ka,kb=str(a.name_key),str(b.name_key)
    if not a['keys'] or not b['keys']: continue
    best=max(((x,y) for x in a["keys"] for y in b["keys"]),key=lambda p:fuzz.ratio(p[0],p[1]))
    ta,tb=set(best[0].split()),set(best[1].split())
    extra=(ta-tb)|(tb-ta); shared=ta&tb
    f=dict(id1=r.id1,id2=r.id2,src1=a.source,src2=b.source,name1=a.name_raw,name2=b.name_raw,
      key_eq=int(bool(a["keys"]&b["keys"])), key_main_eq=int(ka==kb),
      ratio=fuzz.ratio(best[0],best[1])/100, jw=JaroWinkler.similarity(best[0],best[1]),
      tset=fuzz.token_set_ratio(best[0],best[1])/100,
      idf_shared=sum(idf(t) for t in shared), idf_extra=sum(idf(t) for t in extra),
      geo_extra=int(any(t in GEO for t in extra)), n_extra=len(extra),
      emb=float(E[pos[r.id1]]@E[pos[r.id2]]))
    f["country"]= 0 if (pd.isna(a.country) or pd.isna(b.country)) else (1 if a.country==b.country else (0 if {a.country,b.country}<=GREATER_CHINA else -1))
    ca,cb=(None if pd.isna(x) else fold(str(x)).replace(" city","") for x in (a.city,b.city))
    f["city"]= 0 if (ca is None or cb is None) else (1 if ca in cb or cb in ca else -1)
    f["year"]= 0 if (pd.isna(a.founded) or pd.isna(b.founded)) else (1 if abs(int(a.founded[:4])-int(b.founded[:4]))<=1 else -1)
    qs=[max(a[c],b[c])/min(a[c],b[c]) for c in ["revenue","assets"] if pd.notna(a[c]) and pd.notna(b[c]) and a[c]>0 and b[c]>0]
    def rawnum(v):
        try: return None if pd.isna(v) else round(float(str(v).replace(",","").replace(" ","")),2)
        except ValueError: return None
    f["fin_exact"]=int(any(rawnum(a[c+"_raw"]) is not None and rawnum(a[c+"_raw"])==rawnum(b[c+"_raw"]) and rawnum(a[c+"_raw"])>0 for c in ["revenue","assets"]))
    f["fin"]=0 if not qs else (1 if min(qs)<1.6 else (-1 if min(qs)>5 else 0))
    out.append(f)
F=pd.DataFrame(out); F.to_csv(f"{W}/state/features.csv",index=False); print(F.describe().T.to_string())
