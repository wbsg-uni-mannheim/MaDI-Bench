"""Stage 2: normalization. Reads s1 tables, writes work/state/s2_all.pkl with raw_* columns kept."""
import json, os, re
from collections import Counter
import pandas as pd
from normlib import *

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = f"{BASE}/work/state"
SRCS = ["discogs", "lastfm", "musicbrainz"]


def label_list(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [clean_ws(fix_mojibake(x)) for x in s.split("|") if x.strip()]


def main():
    frames = [pd.read_pickle(f"{ST}/s1_{s}.pkl") for s in SRCS]
    df = pd.concat(frames, ignore_index=True).fillna("")
    for c in ["name", "artist", "release-date", "release-country", "duration", "label", "genre", "tracks"]:
        if c not in df:
            df[c] = ""
        df["raw_" + c] = df[c]
    out = pd.DataFrame({"id": df["id"], "source": df["source"]})
    for c in df.columns:
        if c.startswith("raw_"):
            out[c] = df[c]
    out["title"] = [clean_title(t) for t in df["name"]]
    out["title_prefix_artist"] = [title_prefix_artist(t) if s == "lastfm" else None for t, s in zip(df["name"], df["source"])]
    out["artists"] = [clean_artist(a, s) for a, s in zip(df["artist"], df["source"])]
    # lastfm rows missing artist but carrying 'Artist -  Title' prefix: use the prefix as artist evidence
    out["artists"] = [a if a else ([p] if p else []) for a, p in zip(out["artists"], out["title_prefix_artist"])]
    pdts = [parse_date(x) for x in df["release-date"]]
    out["date"] = [p[0] for p in pdts]
    out["year"] = [p[1] for p in pdts]
    out["date_prec"] = [p[2] for p in pdts]
    cc = [norm_country(x) for x in df["release-country"]]
    out["country"] = [c[0] for c in cc]
    out["country_method"] = [c[1] for c in cc]
    out["duration_s"] = [parse_duration(x) for x in df["duration"]]
    out["tracks"] = [parse_tracks(x) for x in df["tracks"]]
    out["labels"] = [label_list(x) for x in df["label"]]
    # genre: atoms, then order by the most frequent ordering seen among fully-clean discogs values
    atoms = [genre_atoms(x) for x in df["genre"]]
    clean_orders = Counter()
    for raw in df.loc[df.source == "discogs", "genre"]:
        parts = raw.split("|") if raw else []
        if parts and all(p in GENRE_ATOMS for p in parts):
            clean_orders[tuple(parts)] += 1
    best_order = {}
    for order, n in clean_orders.most_common():
        best_order.setdefault(frozenset(order), order)
    out["genre"] = ["|".join(best_order.get(frozenset(a), tuple(a))) if a else None for a in atoms]
    out["genre_known_combo"] = [bool(a) and frozenset(a) in best_order for a in atoms]

    # artist display repair: expand initial-abbreviated names ('T. Monochrome Set') when exactly one observed
    # full artist name in any source fits the pattern
    vocab = []
    for a, src in zip(df["artist"], df["source"]):
        if not a.strip():
            continue
        parts = a.split("|") if src == "discogs" else [a]
        for p in parts:
            p = clean_ws(fix_mojibake(p))
            m = re.match(r"^([^,]+),\s*([^,]+)$", p)
            if m and src == "musicbrainz":
                p = f"{m.group(2)} {m.group(1)}"
            vocab.append(p)
    vidx = build_artist_vocab(vocab)
    # names with initials that discogs/musicbrainz list verbatim are genuine ('R. Kelly'): never expanded
    genuine = set(vocab[:0])
    for a, src in zip(df["artist"], df["source"]):
        if src != "lastfm" and a.strip():
            genuine.update(clean_ws(fix_mojibake(p)) for p in a.split("|"))
    out["artist_expanded"] = [expand_initials(clean_ws(fix_mojibake(a)), vidx)
                              if a.strip() and clean_ws(fix_mojibake(a)) not in genuine else None for a in df["artist"]]
    # label spelling repair against frequent observed discogs labels
    lab_counts = Counter(l for ls in out.loc[out.source == "discogs", "labels"] for l in ls)
    fixer = build_label_fixer(lab_counts)
    out["labels_fixed"] = [[fixer.get(l, l) for l in ls] for ls in out["labels"]]
    json.dump(fixer, open(f"{ST}/s2_label_fixes.json", "w"), indent=0, ensure_ascii=False)
    print("artist expansions:", out.artist_expanded.notna().sum(), "label fixes:", len(fixer))

    # comparison keys
    out["k_title"] = out["title"].map(key)
    out["k_artist"] = out["artists"].map(lambda l: " ".join(key(a) for a in l))
    out["k_tracks"] = out["tracks"].map(lambda l: [key(t) for t in l])
    out["k_tracks_base"] = out["tracks"].map(lambda l: [key(re.sub(r"\s*[\(\[].*$", "", t)) or key(t) for t in l])
    out.to_pickle(f"{ST}/s2_all.pkl")

    # diagnostics: parse / canonical rates per source & attribute
    rows = []
    for s, g in out.groupby("source"):
        for attr, raw, val in [("release-date", "raw_release-date", "date"), ("release-country", "raw_release-country", "country"),
                               ("duration", "raw_duration", "duration_s"), ("genre", "raw_genre", "genre"),
                               ("tracks", "raw_tracks", "tracks"), ("name", "raw_name", "title"), ("artist", "raw_artist", "artists")]:
            nn = (g[raw].astype(str).str.strip() != "")
            ok = g[val].map(lambda v: v is not None and v == v and v != [] and v != "")
            rows.append(dict(source=s, attribute=attr, non_null=int(nn.sum()), canonical=int((nn & ok).sum()),
                             unmapped=int((nn & ~ok).sum()), canonical_rate=round((nn & ok).sum() / max(nn.sum(), 1), 4)))
    cov = pd.DataFrame(rows)
    cov.to_csv(f"{BASE}/work/taxonomy_coverage.csv", index=False)
    print(cov.to_string())
    print(out.country_method.value_counts(dropna=False))
    print("genre known-combo rate", out.loc[out.genre.notna(), "genre_known_combo"].mean())


if __name__ == "__main__":
    main()
