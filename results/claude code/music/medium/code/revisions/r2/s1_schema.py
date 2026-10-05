"""Stage 1: schema matching. Map each source's columns to the target schema by meaning
(inspected examples: types, units, formats). Writes sm_mapping.csv and translated tables."""
import pandas as pd, json, os
W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
DATA = f"{ROOT}/task/input/data"
MAP = {  # source -> {source_col: (target_col, score, transformation note)}
 "discogs": {"id": ("id",1.0,"native id"), "name": ("name",1.0,"title"), "artist": ("artist",1.0,"artist, '|' multi, '(n)' disambiguator"),
             "release-date": ("release-date",1.0,"dd.mm.yyyy or ISO -> ISO"), "release-country": ("release-country",1.0,"mixed alias/typo -> canonical name"),
             "duration": ("duration",1.0,"seconds (few mm:ss)"), "label": ("label",1.0,"'|' list"), "genre": ("genre",1.0,"discogs genre, '|' list"),
             "tracks": ("tracks",1.0,"python-list literal")},
 "lastfm": {"id": ("id",1.0,"native id"), "t_nm": ("name",0.95,"title, noisy prefixes ('Artist -  ', '[explicit]', '- new -')"),
            "ar_nm": ("artist",0.95,"artist, sometimes abbreviated initials"), "dur_s": ("duration",1.0,"h:mm:ss -> seconds"),
            "tnms": ("tracks",1.0,"python-list literal, '[]' = unknown")},
 "musicbrainz": {"id": ("id",1.0,"native id"), "rel_nm": ("name",1.0,"title, noisy suffixes '(album)','(orig.)','(release)'"),
                 "art": ("artist",1.0,"'Last, First' inverted, '(and others)'"), "rel_dt": ("release-date",1.0,"dd.mm.yyyy / ISO / yyyy-mm"),
                 "rel_ctry": ("release-country",1.0,"full country names"), "dur": ("duration",1.0,"mm:ss -> seconds; '0:00' = unknown"),
                 "trks": ("tracks",1.0,"python-list literal")},
}
schema = json.load(open(f"{ROOT}/task/input/schemamatching/target_schema.json"))
props = set(schema["properties"])
rows = []
for src, m in MAP.items():
    df = pd.read_csv(f"{DATA}/{src}.csv", dtype=str, keep_default_na=False, na_values=[""])
    assert set(m) == set(df.columns), (src, set(df.columns) ^ set(m))
    for sc, (tc, sco, note) in m.items():
        assert tc in props
        rows.append(dict(source_dataset=src, source_column=sc, target_dataset="music_release", target_column=tc, score=sco, note=note))
    out = df.rename(columns={k: v[0] for k, v in m.items()})
    out.insert(0, "source", src)
    out.to_pickle(f"{W}/state/translated_{src}.pkl")
    missing = sorted(props - set(out.columns))
    print(src, len(out), "targets absent in source:", missing)
mp = pd.DataFrame(rows)
mp.to_csv(f"{W}/state/sm_mapping_annotated.csv", index=False)
os.makedirs(f"{ROOT}/submission", exist_ok=True)
mp.drop(columns="note").to_csv(f"{ROOT}/submission/sm_mapping.csv", index=False)
