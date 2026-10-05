"""Stage 2: normalization. Builds work/state/records.pkl: one row per source record,
raw values kept (raw_*), canonical values for fusion, comparison keys for matching."""
import pandas as pd, re, ast, json, unicodedata
SRC = {
 "discogs": dict(id="rec_uid", name="title_str", artist="performer", date="pub_dt", country="origin_loc",
                 duration="duration", label="imprint", genre="category", tracks="tracks_track-name"),
 "lastfm": dict(id="item_code", name="album_title", artist="band", duration="album_length", tracks="tracks_track-name"),
 "musicbrainz": dict(id="Attribute_1", name="Attribute_2", artist="Attribute_3", date="Attribute_4",
                     country="Attribute_5", duration="Attribute_6", tracks="Attribute_9"),
}
COUNTRY = {"UK": "United Kingdom of Great Britain and Northern Ireland", "Russia": "Russian Federation",
           "South Korea": "Korea, Republic of", "Bolivia": "Bolivia, Plurinational State of",
           "Bosnia & Herzegovina": "Bosnia and Herzegovina"}
GENRE = {"Rock": "Rock", "Electronic": "Electronic / Dance", "Pop": "Pop", "Funk / Soul": "R&B / Soul",
         "Hip Hop": "Hip Hop / Rap", "Jazz": "Jazz", "Classical": "Classical", "Reggae": "Reggae / Dub",
         "Blues": "Blues", "Stage & Screen": "Soundtrack / Score", "Latin": "World Music",
         "Folk, World, & Country": "Folk"}   # Non-Music, Children's, Brass & Military: unmapped
TAX = set(pd.read_csv("task/input/schemamatching/Music_Genres_Taxonomy.csv")["Genre Name"])
assert set(GENRE.values()) <= TAX

def fix_moji(s):
    for _ in range(3):
        if not re.search(r"[ÃÂâ]", s): break
        try: s2 = s.encode("cp1252").decode("utf8")
        except Exception:
            try: s2 = s.encode("latin1").decode("utf8")
            except Exception: break
        if s2 == s: break
        s = s2
    return s

def ws(s): return re.sub(r"\s+", " ", s).strip()

def key(s):
    s = unicodedata.normalize("NFKD", fix_moji(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", " and ")
    return ws(re.sub(r"[^a-z0-9]+", " ", s))

LF_PREFIX = re.compile(r"^(\*popular\*|\[explicit\]|album:|- new -)\s*", re.I)
def clean_name(src, name, artist):
    s = fix_moji(name)
    if src == "lastfm":
        for _ in range(2): s = LF_PREFIX.sub("", s)
        m = re.match(r"^(.+?) -  (.+)$", s)          # 'Artist -  Title' (double space) prefix
        if m and key(m.group(1)) == key(artist): s = m.group(2)
    if src == "musicbrainz":
        s = re.sub(r"\s*\((orig\.|album|release)\)\s*$", "", s)
    return ws(s)

def clean_artist(src, a):
    a = ws(fix_moji(a))
    if src == "musicbrainz":   # injected noise: ' (and others)' / trailing ' feat.'; 'Last, First' inversion
        a = ws(re.sub(r"\s*\(and others\)|\s+feat\.?$", "", a))
        m = re.match(r"^([^,]+), ([^,]+)$", a)
        if m: a = f"{m.group(2)} {m.group(1)}"
    return a

def parse_list(x):
    x = x.strip()
    if not x: return []
    try:
        v = ast.literal_eval(x)
        return [ws(fix_moji(str(t))) for t in v if str(t).strip()]
    except Exception:
        return None

def norm_date(x):
    x = x.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x): return x, "day"
    if re.fullmatch(r"\d{4}-\d{2}", x): return x, "month"
    if re.fullmatch(r"\d{4}", x): return x, "year"
    return "", ("empty" if not x else "fail")

rows = []; stats = []
for src, c in SRC.items():
    df = pd.read_csv(f"task/input/data/{src}.csv", dtype=str, keep_default_na=False)
    fails = 0
    for _, r in df.iterrows():
        g = lambda k: r[c[k]] if k in c else ""
        tr = parse_list(g("tracks"))
        if tr is None: fails += 1; tr = []
        name = clean_name(src, g("name"), g("artist"))
        artist = clean_artist(src, g("artist"))
        date, prec = norm_date(g("date"))
        dur = g("duration").strip()
        dur = int(float(dur)) if re.fullmatch(r"\d+(\.\d+)?", dur) and float(dur) > 0 else None
        genres = [x for x in g("genre").split("|") if x] if src == "discogs" else []
        gcanon = next((GENRE[x] for x in genres if x in GENRE), "")
        country = ws(g("country")); country = COUNTRY.get(country, country)
        labels = [ws(x) for x in g("label").split("|") if ws(x)]
        rows.append(dict(id=r[c["id"]], source=src,
            raw_name=g("name"), raw_artist=g("artist"), raw_date=g("date"), raw_country=g("country"),
            raw_duration=g("duration"), raw_label=g("label"), raw_genre=g("genre"),
            name=name, artist=artist, date=date, date_prec=prec, country=country, duration=dur,
            labels=labels, genre=gcanon, tracks=tr,
            k_name=key(name), k_artist=key(artist), k_tracks=[key(t) for t in tr]))
    stats.append(dict(source=src, n=len(df), track_parse_fail=fails))
R = pd.DataFrame(rows)
assert R.id.is_unique
R.to_pickle("work/state/records.pkl")
# taxonomy coverage
cov = []
d = R[R.source == "discogs"]
cov.append(dict(source="discogs", attribute="genre", non_null=int((d.raw_genre != "").sum()),
                canonical=int((d.genre != "").sum()), unmapped=int(((d.raw_genre != "") & (d.genre == "")).sum())))
for x in cov: x["canonical_rate"] = round(x["canonical"] / x["non_null"], 4)
pd.DataFrame(cov).to_csv("work/taxonomy_coverage.csv", index=False)
json.dump(dict(genre=dict(path="task/input/schemamatching/Music_Genres_Taxonomy.csv", column="Genre Name",
    exhaustive=True, alias_map=GENRE, policy="first Discogs genre in source order that has a mapping; "
    "Non-Music/Children's/Brass & Military left missing (no defensible taxonomy genre)"),
    country=dict(exhaustive=False, alias_map=COUNTRY, policy="discogs short names -> MusicBrainz full names; market labels kept")),
    open("work/taxonomy_plan.json", "w"), indent=1)
print(pd.DataFrame(stats)); print(cov)
print(R.groupby("source").apply(lambda g: pd.Series(dict(date_prec=g.date_prec.value_counts().to_dict(),
      dur=g.duration.notna().mean(), tracks=(g.tracks.str.len() > 0).mean()))).to_string())
