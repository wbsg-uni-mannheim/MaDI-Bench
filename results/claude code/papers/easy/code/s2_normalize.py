"""Stage 2: normalization. Writes work/state/norm_<src>.pkl with raw columns kept (raw_*) and
canonical columns. Same functions applied to every source."""
import pandas as pd, numpy as np, re, ast, html, unicodedata, json, os
W=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(W); ST=f'{W}/state'
os.makedirs(ST,exist_ok=True)
SRCS=['crossref','dblp','open_alex']
TYPE_ENUM=["article","inproceedings","incollection","posted-content","review","editorial","letter","preprint","erratum","book-chapter","paratext"]
TYPE_ALIAS={'journal-article':'article','journal article':'article','journal_article':'article','book_chapter':'book-chapter','proceedings-article':'inproceedings'}
TAG=re.compile(r'<[^>]*>?')
def clean_title(t, src):
    if not isinstance(t,str): return None
    t=html.unescape(html.unescape(t))
    t=TAG.sub('',t)                       # strip html/mathml tags, keep inner text
    t=re.sub(r'[​-‏‪-‮]','',t)
    t=re.sub(r'\s+',' ',t).strip()
    if src=='dblp' and t.endswith('.') and not t.endswith('..'): t=t[:-1].rstrip()  # dblp convention adds final period
    return t or None
def fold(s):
    s=unicodedata.normalize('NFKD',s)
    s=''.join(c for c in s if not unicodedata.combining(c))
    return s.lower()
def title_key(t):
    return re.sub(r'[^a-z0-9]+',' ',fold(t)).strip() if t else ''
AFFIL=re.compile(r'\d|universit|institut|department|college|school|laborator|chairman|consultant|academy|research|center|centre|hospital|\bltd\b|\binc\b|corporation',re.I)
def clean_authors(a, src):
    if not isinstance(a,str): return None
    try: L=ast.literal_eval(a)
    except Exception: return None
    out=[]
    for x in L:
        x=re.sub(r'[​-‏‪-‮]','',str(x)); x=re.sub(r'\s+',' ',x).strip()
        if src=='dblp': x=re.sub(r'\s\d{4}$','',x)
        if not x: continue
        out.append(x)
    return out or None
def author_keys(L):
    """surname-ish tokens for matching: set of all folded alpha tokens of len>1 per author"""
    if not L: return frozenset()
    toks=set()
    for x in L:
        if AFFIL.search(x): continue
        for w in re.findall(r'[a-z]+',fold(x)):
            if len(w)>1: toks.add(w)
    return frozenset(toks)
def clean_num(v):
    """integer-like fields with injected spaces ('2 018', '12. 0', '5 0.0')"""
    if not isinstance(v,str): return None
    s=v.strip()
    if re.fullmatch(r'\d{1,3}(,\d{3})+(\.0+)?',s): s=s.replace(',','')
    if re.fullmatch(r'[\d .]+',s):
        s=s.replace(' ','')
        if re.fullmatch(r'\d+(\.0+)?',s): return str(int(float(s)))
        return None
    return None
def clean_code(v):
    """volume/issue/pages: numeric noise fixed, alnum locators kept"""
    if not isinstance(v,str): return None
    n=clean_num(v)
    if n is not None: return n
    s=re.sub(r'\s+',' ',v.strip())
    return s or None
def clean_journal(j):
    if not isinstance(j,str): return None
    j=re.sub(r'\s+',' ',html.unescape(html.unescape(j))).strip()
    return j or None
def main():
    cov=[]
    for src in SRCS:
        d=pd.read_csv(f'{R}/task/input/data/{src}.csv',dtype=str)
        o=pd.DataFrame({'id':d.id,'source':src})
        for c in d.columns:
            if c!='id': o['raw_'+c]=d[c]
        tl=d.type.str.strip().str.lower().map(lambda x: TYPE_ALIAS.get(x,x) if isinstance(x,str) else None)
        o['type']=tl.where(tl.isin(TYPE_ENUM))
        o['title']=d.title.map(lambda t: clean_title(t,src))
        o['tkey']=o.title.map(title_key)
        o['authors']=d.authors.map(lambda a: clean_authors(a,src))
        o['akeys']=o.authors.map(author_keys)
        o['year']=pd.to_numeric(d.publication_year.map(clean_num),errors='coerce').astype('Int64')
        o['journal']=d.journal.map(clean_journal)
        for c in ['volume','issue','first_page','last_page']: o[c]=d[c].map(clean_code)
        for c in ['referenced_works_count','cited_by_count']: o[c]=pd.to_numeric(d[c].map(clean_num),errors='coerce').astype('Int64')
        for c,raw in [('type','type'),('title','title'),('authors','authors'),('year','publication_year'),('journal','journal'),('volume','volume'),('issue','issue'),('first_page','first_page'),('last_page','last_page'),('referenced_works_count','referenced_works_count'),('cited_by_count','cited_by_count')]:
            nn=int(d[raw].notna().sum()); ok=int(o[c].notna().sum())
            cov.append(dict(source=src,attribute=c,non_null=nn,canonical=ok,unmapped=nn-ok,canonical_rate=round(ok/max(nn,1),4)))
        o.to_pickle(f'{ST}/norm_{src}.pkl')
        print(src,len(o))
    cv=pd.DataFrame(cov); cv.to_csv(f'{W}/taxonomy_coverage.csv',index=False); print(cv[cv.unmapped>0].to_string())
if __name__=='__main__': main()
