"""Stage 2: translate each source to target attributes + normalized comparison columns."""
import pandas as pd, json, re, sys
sys.path.insert(0,'work')
from norm import *
D='task/input/data/'
def year(v):
    m = re.match(r"^(\d{4})", str(v)) if isinstance(v,str) else None
    return int(m.group(1)) if m else None

db = pd.read_csv(D+'dbpedia.csv', dtype=str)
mc = pd.read_csv(D+'metacritic.csv', dtype=str)
sa = pd.read_csv(D+'sales.csv', dtype=str)
rows=[]
for r in db.itertuples():
    t, dy, tag = strip_disambig(r.title)
    fr = strip_disambig(r.franchise)[0] if isinstance(r.franchise,str) else None
    rows.append(dict(record_id=r.wiki_ref, source='dbpedia', name=t, name_raw=r.title, disamb_year=dy, disamb_tag=tag,
        year=year(r.launch_yr), date_raw=r.launch_yr, developer=r.studio, publisher=None,
        platform_raw=r.system, genres_raw=r.genre, series=fr, criticScore=None, userScore=None, ESRB=None))
for r in mc.itertuples():
    rows.append(dict(record_id=r.mc_id, source='metacritic', name=strip_disambig(r.game_title)[0], disamb_year=strip_disambig(r.game_title)[1], name_raw=r.game_title,
        year=year(r.year_published), date_raw=r.year_published, developer=r.made_by, publisher=None,
        platform_raw=r.console, genres_raw=r.genres, series=None,
        criticScore=float(r.press_rating) if isinstance(r.press_rating,str) else None,
        userScore=float(r.player_rating) if isinstance(r.player_rating,str) else None, ESRB=esrb(r.age_rating)))
for r in sa.itertuples():
    rows.append(dict(record_id=r.rec_id, source='sales', name=strip_disambig(r.prod_title)[0], disamb_year=strip_disambig(r.prod_title)[1], name_raw=r.prod_title,
        year=year(r.launch_dt), date_raw=r.launch_dt, developer=r.studio, publisher=r.dist,
        platform_raw=r.hw, genres_raw=r.genre, series=None,
        criticScore=float(r.press_score) if isinstance(r.press_score,str) else None,
        userScore=float(r.comm_rating) if isinstance(r.comm_rating,str) else None, ESRB=esrb(r.age_classification)))
N = pd.DataFrame(rows)
N['pkey'] = N.platform_raw.map(platform_key)
N['tkey'] = N.name.map(title_key)
N.to_pickle('work/state/normalized.pkl')
N.drop(columns=[]).to_csv('work/state/normalized.csv', index=False)
# taxonomy / parse coverage diagnostics
cov=[]
for s,g in N.groupby('source'):
    cov.append(dict(source=s, attribute='platform', non_null=int(g.platform_raw.notna().sum()),
        canonical=int(g.pkey.notna().sum() - g.pkey.fillna('').str.startswith('other:').sum()),
        unmapped=int(g.pkey.fillna('').str.startswith('other:').sum())))
    cov.append(dict(source=s, attribute='year', non_null=int(g.date_raw.notna().sum()), canonical=int(g.year.notna().sum()),
        unmapped=int(g.date_raw.notna().sum()-g.year.notna().sum())))
    cov.append(dict(source=s, attribute='ESRB', non_null=int(g.ESRB.notna().sum()), canonical=int(g.ESRB.notna().sum()), unmapped=0))
C=pd.DataFrame(cov); C['canonical_rate']=(C.canonical/C.non_null.clip(lower=1)).round(3)
C.to_csv('work/taxonomy_coverage.csv', index=False); print(C)
print(N.groupby('source').pkey.value_counts().groupby(level=0).head(25))
