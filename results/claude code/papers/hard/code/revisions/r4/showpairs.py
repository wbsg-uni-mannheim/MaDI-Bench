import pandas as pd, sys
sys.path.insert(0, 'work')
from show import show
F = ['t_max','t_min','a_min','a_max','a_first','j_sim','y_diff','vo_eq','is_eq','fi_eq','la_eq']
def sp(df, k=8, seed=0):
    d = df.sample(min(k, len(df)), random_state=seed)
    for _, r in d.iterrows():
        print(' '.join(f'{f}={r[f]:.2f}' if isinstance(r[f], float) else f'{f}={r[f]}' for f in F if pd.notna(r[f])))
        show([r.id1, r.id2]); print()
