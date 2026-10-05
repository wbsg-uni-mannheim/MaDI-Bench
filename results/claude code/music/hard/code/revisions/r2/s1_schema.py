"""Stage 1: schema matching. Maps each source's columns onto target attributes
and writes translated (still raw-valued) tables to work/state/s1_<source>.pkl."""
import json, pandas as pd, os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{BASE}/task/input/data"
ST = f"{BASE}/work/state"

# (source column -> target attribute, confidence, note)
MAPPING = {
    "discogs": {"id": ("id", 1.0, "native id"), "rel_nm": ("name", 1.0, "release title"),
                "art_nm": ("artist", 1.0, "artist, '|' separated, discogs '(n)' disambiguators"),
                "rel_dt": ("release-date", 1.0, "mixed date formats"),
                "rel_ctry": ("release-country", 1.0, "country, abbreviations/scrambles"),
                "dur": ("duration", 1.0, "mixed h:m:s / 'Xm Ys' / seconds"),
                "lbl": ("label", 1.0, "'|' separated labels"),
                "gnr": ("genre", 1.0, "discogs genre vocabulary, '|' separated"),
                "trks": ("tracks", 1.0, "python list literal")},
    "lastfm": {"id": ("id", 1.0, "native id"),
               "Attribute_2": ("name", 0.95, "title; sometimes prefixed 'Artist -  ' or promo tags"),
               "Attribute_3": ("artist", 0.95, "artist name, often abbreviated"),
               "Attribute_6": ("duration", 0.9, "duration formats like '12m 14s', '15:03', seconds"),
               "Attribute_9": ("tracks", 0.95, "python list literal of track titles")},
    "musicbrainz": {"id": ("id", 1.0, "native id"), "t": ("name", 1.0, "title with '(orig.)'-like suffix noise"),
                    "ar": ("artist", 1.0, "'Last, First' ordering frequent"),
                    "rd": ("release-date", 1.0, "mixed date formats"),
                    "rc": ("release-country", 1.0, "full country names, scrambles"),
                    "du": ("duration", 1.0, "duration formats"),
                    "tk": ("tracks", 1.0, "python list literal")},
}

def main():
    rows = []
    for src, mp in MAPPING.items():
        df = pd.read_csv(f"{DATA}/{src}.csv", dtype=str, keep_default_na=False)
        assert set(mp) <= set(df.columns), (src, set(mp) - set(df.columns))
        unm = set(df.columns) - set(mp)
        assert not unm, (src, unm)
        out = pd.DataFrame({tgt: df[col] for col, (tgt, _, _) in mp.items()})
        out.insert(0, "source", src)
        out.to_pickle(f"{ST}/s1_{src}.pkl")
        for col, (tgt, sc, note) in mp.items():
            rows.append(dict(source_dataset=src, source_column=col, target_dataset="music_release",
                             target_column=tgt, score=sc, note=note))
    sm = pd.DataFrame(rows)
    sm.to_csv(f"{ST}/sm_mapping_annotated.csv", index=False)
    os.makedirs(f"{BASE}/submission", exist_ok=True)
    sm.drop(columns="note").to_csv(f"{BASE}/submission/sm_mapping.csv", index=False)
    print(sm.to_string())

if __name__ == "__main__":
    main()
