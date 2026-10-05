"""Stage 2: normalization into a canonical long table (raw values kept alongside)."""
from urllib.parse import unquote
import os, re, json, ast, unicodedata, pandas as pd, numpy as np
from aliases import COUNTRY, FORBES_IND, DBP_IND, LEGAL
W=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(W)
TAX=f"{ROOT}/task/input/schemamatching"
cl=pd.read_csv(f"{TAX}/CLDR_Country_Taxonomy.csv")
CANON={}
for _,r in cl.iterrows():
    for col in ["Country Name","Country Short Name","Country Variant Name"]:
        if isinstance(r[col],str): CANON[r[col].strip().lower()]=r["Country Name"]
for k,v in COUNTRY.items():
    assert v.lower() in CANON, v
    CANON[k]=CANON[v.lower()]
GICS=set(pd.read_csv(f"{TAX}/GICS_Industry_Taxonomy.csv")["Industry Name"])
for d in (FORBES_IND,DBP_IND):
    for v in d.values(): assert v in GICS, v
NULLS={"<na>","<n a>","na","n/a","nan","none","null",""}

def isnull(v): return v is None or (isinstance(v,float) and np.isnan(v)) or str(v).strip().lower() in NULLS
def country(v):
    if isnull(v): return None
    return CANON.get(str(v).strip().lower())
def fold(s):
    s=unicodedata.normalize("NFKD",s); s="".join(c for c in s if not unicodedata.combining(c))
    return s.lower()
def name_key(s):
    """matching key: accents folded, parentheticals, punctuation, legal/generic suffix tokens removed"""
    s=fold(s); s=re.sub(r"\(.*?\)"," ",s); s=s.replace("&"," and ").replace("+"," and ")
    s=re.sub(r"[.'`’]","",s); s=re.sub(r"[\W_]+"," ",s)
    toks=s.split(); core=[t for t in toks if t not in LEGAL]
    return " ".join(core) if core else " ".join(toks)
SUFFIX=re.compile(r"(,?\s+(inc|incorporated|corp|corporation|co|ltd|limited|plc|llc|l\.l\.c|s\.?a|a\.?g|se|n\.?v|b\.?v|gmbh|sarl|sas|s\.p\.a|spa|pte|pty|oyj|asa|ab|bhd|berhad|tbk)\.?)+\s*$",re.I)
def display_name(s, src):
    s=str(s).strip()
    if src=="dbpedia": s=re.sub(r"\s*\((company|companies|corporation|business|brand|enterprise|retailer|firm|conglomerate|bank|[a-z][^)]*)\)\s*$","",s)
    s=re.sub(r"\s*\((pte|pty|del|the)\)\s*"," ",s,flags=re.I).strip()
    prev=None
    while prev!=s: prev=s; s=SUFFIX.sub("",s).strip().rstrip(",").strip()
    return s or None
def founded(v):
    if isnull(v): return None
    v=str(v).strip()
    m=re.match(r"^(\d{4})-(\d{2})-(\d{2})$",v)
    if m: y,mo,d=m.groups()
    else:
        m=re.match(r"^([A-Za-z]+) (\d{1,2}), (\d{4})$",v)
        if not m: return "PARSEFAIL"
        mon=["january","february","march","april","may","june","july","august","september","october","november","december"].index(m.group(1).lower())+1
        y,mo,d=m.group(3),f"{mon:02d}",f"{int(m.group(2)):02d}"
    if not (1700<=int(y)<=2016): return "OUTOFRANGE"
    return f"{y}-{mo}-{d}"
def num(v):
    if isnull(v): return None
    try: return float(str(v).replace(",","").replace(" ",""))
    except: return "PARSEFAIL"
def dbp_money(v):
    x=num(v)
    if x is None or x=="PARSEFAIL": return x
    if x<100: return int(round(x*1e9))
    if x<1e4 and re.match(r"^\d{1,3}(,\d{3})*\.\d{2}$",str(v).strip()): return int(round(x*1e9))  # Forbes-style 'bn' string
    if x<1e5: return int(round(x*1e6))
    return int(round(x))
def people(v):
    if isnull(v): return None
    v=str(v).strip()
    items=ast.literal_eval(v) if v.startswith("[") else [v]
    out=[]
    for it in items:
        it=re.sub(r"\s*\([a-z][^)]*\)\s*$","",str(it)).strip()
        if it and it.lower() not in NULLS and it not in out: out.append(it)
    return json.dumps(out,ensure_ascii=False) if out else None
def city(v):
    return None if isnull(v) else re.sub(r"\s+"," ",str(v)).strip()

rows=[]; cov=[]
for src in ["dbpedia","forbes","fullcontact"]:
    d=pd.read_csv(f"{ROOT}/task/input/data/{src}.csv",dtype=str,keep_default_na=False,na_values=[""])
    for _,r in d.iterrows():
        g=lambda c: r[c] if c in d.columns else None
        nm=r["name"]
        if src=="dbpedia":
            slug=unquote(r["id"].split("/resource/",1)[1]).replace("_"," ")
            if isnull(nm) or "\ufffd" in str(nm) or "?" in str(nm): nm=slug   # repair mojibake/missing names from the URI
        o=dict(record_id=r["id"],source=src,name_raw=r["name"],name=None if isnull(nm) else display_name(nm,src),name_key=None if isnull(nm) else name_key(str(nm)),
               country_raw=g("country"),country=country(g("country")),city=city(g("city")),founded=founded(g("founded")),
               keypeople=people(g("keypeople")))
        ind=g("industry")
        o["industry_raw"]=ind
        o["industry"]=None if isnull(ind) else (FORBES_IND if src=="forbes" else DBP_IND).get(str(ind).strip())
        if src=="forbes":
            for c in ["assets","revenue"]:
                x=num(r[c]); o[c]=None if x is None else (x if x=="PARSEFAIL" else int(round(x*1e9)))
            o["alt_key"]=name_key(r["id"].rstrip("/").split("/")[-1].replace("-"," "))
        else:
            for c in ["assets","revenue"]: o[c]=dbp_money(g(c)) if src=="dbpedia" else None
            if src=="dbpedia":
                sk=name_key(slug); o["alt_key"]=sk if sk!=o["name_key"] else None
                o["assets_raw"]=g("assets"); o["revenue_raw"]=g("revenue"); rows.append(o); continue
            parts=re.split(r"\s+[-–|:]\s+|\s*\|\s*",str(r["name"]))
            o["alt_key"]="|".join(dict.fromkeys(k for k in (name_key(p) for p in parts) if k)) if len(parts)>1 else None
        o["assets_raw"]=g("assets"); o["revenue_raw"]=g("revenue")
        rows.append(o)
N=pd.DataFrame(rows)
# taxonomy coverage
for src,g in N.groupby("source"):
    for a,raw in [("country","country_raw"),("industry","industry_raw")]:
        nn=g[raw].notna().sum(); ok=g[a].notna().sum()
        cov.append(dict(source=src,attribute=a,non_null=int(nn),canonical=int(ok),unmapped=int(nn-ok),canonical_rate=round(ok/nn,4) if nn else None))
    for a in ["founded","assets","revenue"]:
        cov.append(dict(source=src,attribute=a+"_parse_fail",non_null=int(g[a].notna().sum()),canonical=int(g[a].isin(["PARSEFAIL","OUTOFRANGE"]).sum()),unmapped=None,canonical_rate=None))
for a in ["founded","assets","revenue"]:
    N.loc[N[a].isin(["PARSEFAIL","OUTOFRANGE"]),a]=None
pd.DataFrame(cov).to_csv(f"{W}/taxonomy_coverage.csv",index=False)
N.to_csv(f"{W}/state/normalized.csv",index=False)
json.dump({"country":{"path":"task/input/schemamatching/CLDR_Country_Taxonomy.csv","canonical":"Country Name","aliases":["Country Short Name","Country Variant Name","work/aliases.py:COUNTRY"],"exhaustive":True,"unmapped_policy":"null (historical states etc.)"},
 "industry":{"path":"task/input/schemamatching/GICS_Industry_Taxonomy.csv","canonical":"Industry Name","aliases":["work/aliases.py:FORBES_IND","work/aliases.py:DBP_IND"],"exhaustive":True,"unmapped_policy":"null"},
 "keypeople":{"serialization":"JSON list"}},open(f"{W}/taxonomy_plan.json","w"),indent=1)
print(pd.DataFrame(cov).to_string()); print(N.shape)
