import pandas as pd, sys
df=pd.read_pickle('state/s2_all.pkl').set_index('id')
c=pd.read_pickle('state/s5_scored.pkl')
def show(i):
    r=df.loc[i]; return f"{i}: {r.raw_name[:50]!r} | {r.raw_artist[:30]!r} | {r['raw_release-date']} | {r.country} | {r.duration_s} | {r.raw_tracks[:100]}"
def pairs(sub, n=15, seed=0):
    for _,p in sub.sample(min(n,len(sub)),random_state=seed).iterrows():
        print(f"score={p.score} t={p.t_sim:.2f}/{p.t_strict:.2f} a={p.a_sim:.2f} tr={p.tr_short:.2f}/{p.tr_n} dur={p.dur_rel:.2f} yd={p.year_diff} ce={p.country_eq}")
        print('   ',show(p.id1)); print('   ',show(p.id2))
