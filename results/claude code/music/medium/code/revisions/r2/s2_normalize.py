"""Stage 2: normalization. Symmetric canonical representations for all sources.
Raw values are kept in *_raw columns; comparison keys are separate from display values."""
import pandas as pd, numpy as np, re, ast, unicodedata, json, os
from collections import Counter
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

W = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(W)
SRCS = ["discogs", "lastfm", "musicbrainz"]

# ---------------- text helpers ----------------
MOJI = re.compile(r"[ÃÂâ][\x80-\xbfŒœŠšŸŽžƒˆ˜–-›€™]|Ã.|â€")

def fix_mojibake(s):
    if not isinstance(s, str):
        return s
    for _ in range(3):
        if not MOJI.search(s):
            break
        fixed = None
        for enc in ("cp1252", "latin-1"):
            try:
                fixed = s.encode(enc).decode("utf-8"); break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        if fixed is None:  # partial: fix char-run by char-run
            try:
                fixed = re.sub(r"[À-ÿ][\u0080-¿Œ-™]+",
                               lambda m: m.group(0).encode("cp1252", "ignore").decode("utf-8", "ignore") or m.group(0), s)
            except Exception:
                break
        if fixed == s:
            break
        s = fixed
    return s

def ascii_fold(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

def text_key(s):
    """lowercase ascii alnum tokens; & -> and"""
    if not isinstance(s, str):
        return ""
    s = ascii_fold(s).lower().replace("&", " and ")
    s = re.sub(r"(?<=\w)['`’](?=\w)", "", s)   # fermat's -> fermats
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def collapse_ws(s):
    return " ".join(s.split()) if isinstance(s, str) else s

# ---------------- names ----------------
LFM_PREFIX = re.compile(r"^\s*(\[explicit\]|- new -|Album:|\*popular\*)\s*", re.I)
MB_SUFFIX = re.compile(r"\s*\((album|orig\.|release)\)\s*$", re.I)

def clean_name(src, s, artist):
    s = fix_mojibake(s)
    note = []
    if src == "lastfm":
        while True:
            m = LFM_PREFIX.match(s)
            if not m: break
            s = s[m.end():]; note.append("prefix:" + m.group(1))
        m = re.match(r"^(.+?) -  (.+)$", s)  # 'Artist -  Title' (double space is the lastfm artifact)
        if m:
            s = m.group(2); note.append("artist_prefix")
    if src == "musicbrainz":
        m = MB_SUFFIX.search(s)
        if m:
            s = s[:m.start()]; note.append("suffix:" + m.group(1))
    return collapse_ws(s), "|".join(note)

# ---------------- artists ----------------
def clean_artist(src, s):
    """returns display string and list of artist-name variants"""
    if not isinstance(s, str) or not s.strip():
        return None
    s = fix_mojibake(s)
    parts = s.split("|") if src == "discogs" else [s]
    out = []
    for p in parts:
        p = re.sub(r"\s*\(\d+\)\s*$", "", p)            # discogs disambiguator
        p = re.sub(r"\s*\(and\s*(\w+\s+)?others\)\s*$", "", p)  # mb '(and others)'
        if src == "musicbrainz":
            p = re.sub(r"\s+feat\.?$", "", p, flags=re.I)   # mb dangling 'feat.' artifact
        if src == "lastfm":
            p = re.sub(r"\s*\(\.$|\s+\.$", "", p)          # lastfm dangling ' .' / ' (.' artifact
        p = collapse_ws(p)
        if src == "musicbrainz" or (src == "discogs" and "," in p):
            segs = [x.strip() for x in p.split(",")]
            segs = [x for x in segs if x]
            if len(segs) >= 2 and not re.search(r"\d", p) and not (len(segs) == 2 and segs[1].lower() in ("jr.", "jr", "sr.")):
                # 'Last, First[, Rest][, The]'  -> 'The First Last Rest'
                the = [x for x in segs[1:] if x.lower() == "the"]
                rest = [x for x in segs[1:] if x.lower() != "the"]
                if rest:
                    p = " ".join((["The"] if the else []) + [rest[0], segs[0]] + rest[1:])
                else:
                    p = " ".join((["The"] if the else []) + [segs[0]])
        out.append(p)
    return " | ".join(out) if len(out) > 1 else out[0]

# ---------------- dates ----------------
OCR = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "i": "1", "B": "8", "G": "6", "q": "9"})

def parse_date(s):
    """-> (iso_full or None, year or None, month or None, status)"""
    if not isinstance(s, str) or not s.strip():
        return None, None, None, "missing"
    raw = s
    s = s.replace(" ", "").translate(OCR)
    status = "ok" if s == raw.replace(" ", "") else "ocr_fixed"
    if " " in raw: status = "ocr_fixed"
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", s)
    if m:
        d_, mo, y = m.groups()
    else:
        digits = s.replace("-", "")
        if not re.fullmatch(r"\d+", digits):
            y = re.match(r"(\d{4})", s)
            return None, (int(y.group(1)) if y else None), None, "unparsed"
        if len(digits) == 8:
            y, mo, d_ = digits[:4], digits[4:6], digits[6:]
        elif len(digits) == 6:
            y, mo, d_ = digits[:4], digits[4:6], None
        elif len(digits) == 4:
            y, mo, d_ = digits, None, None
        else:
            return None, None, None, "unparsed"
    y = int(y)
    if not (1900 <= y <= 2030):
        return None, None, None, "unparsed"
    if mo is not None and not (1 <= int(mo) <= 12):
        return None, y, None, "bad_month"
    if d_ is None:
        return None, y, (int(mo) if mo else None), "partial"
    try:
        iso = pd.Timestamp(year=y, month=int(mo), day=int(d_)).strftime("%Y-%m-%d")
    except ValueError:
        return None, y, int(mo), "bad_day"
    return iso, y, int(mo), status

# ---------------- duration ----------------
def parse_duration(src, s):
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        v = int(s)
    else:
        p = s.split(":")
        if not all(re.fullmatch(r"\d+", x) for x in p):
            return None
        p = [int(x) for x in p]
        v = p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]
    return v if v > 0 else None   # '0:00' in musicbrainz = unknown

# ---------------- tracks ----------------
def parse_tracks(s):
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        v = ast.literal_eval(s)
    except Exception:
        return None
    v = [collapse_ws(fix_mojibake(str(x))) for x in v if str(x).strip()]
    return v or None

def track_core(t):
    """comparison-only core title: drop bracketed qualifiers and ' - ...' suffixes (versions, guests, sessions)"""
    c = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", " ", t)
    c = re.split(r"\s+-\s+", c)[0]
    c = re.sub(r"\s+(feat\.?|featuring|ft\.)\s.*$", "", c, flags=re.I)
    return text_key(c) or text_key(t)

# ---------------- country ----------------
UK = "United Kingdom of Great Britain and Northern Ireland"; US = "United States of America"
ALIASES = {"uk": UK, "u k": UK, "united kingdom": UK, "great britain": UK, "gb": UK, "england": UK,
           "usa": US, "us": US, "usa of america": US, "united states": US, "america": US,
           "nippon": "Japan", "deutschland": "Germany", "bundesrepublik deutschland": "Germany",
           "federal republic of germany": "Germany",
           "republique francaise": "France", "espana": "Spain", "brasil": "Brazil", "holland": "Netherlands",
           "hellas": "Greece", "suomi": "Finland", "eire": "Ireland", "aotearoa": "New Zealand",
           "magyarorszag": "Hungary", "ceska republika": "Czech Republic", "commonwealth of australia": "Australia",
           "canada federal country": "Canada", "jamaica country": "Jamaica", "republic of poland": "Poland",
           "kingdom of belgium": "Belgium", "republic of belarus": "Belarus", "republic of the philippines": "Philippines",
           "portugal republica portuguesa": "Portugal", "korea republic of": "South Korea", "hk": "Hong Kong",
           "jp": "Japan", "ukraina": "Ukraine", "bulgariya": "Bulgaria", "zealand": "New Zealand",
           "united states of": US, "states of america": US, "united of america": US}
CYR = {"Украина": "Ukraine"}
# comparison key groups (different naming conventions of the same country)
CKEY = {"Russian Federation": "Russia", "Korea, Republic of": "South Korea", "Moldova, Republic of": "Moldova",
        "Bolivia, Plurinational State of": "Bolivia"}

def build_country_vocab(values):
    cnt = Counter(v for v in values if isinstance(v, str))
    vocab = set()
    for v, c in cnt.items():
        if c >= 5 and text_key(v) not in ALIASES and v == v.strip() and "  " not in v:
            # exclude obviously corrupted spellings: must be (near) Title case words
            if re.fullmatch(r"[A-Z][a-z]+([ ,&().'-]+(of|and|the|[A-Z][a-z]*|&|UK|US|USA))*[)]?", v) or v in (UK, US):
                vocab.add(v)
    vocab |= {v for v, c in cnt.items() if c >= 20 and re.fullmatch(r"[A-Za-z ,&]+", v) and text_key(v) not in ALIASES}
    vocab |= {UK, US, "Russian Federation", "Russia", "South Korea", "New Zealand", "Hong Kong"}
    # word-shuffled corruptions can be frequent enough to pass the count filter: per sorted-token key keep
    # only the most frequent spelling (canonical UK/US names always win)
    best = {}
    for v in vocab:
        key = " ".join(sorted(text_key(v).split()))
        c = 10**9 if v in (UK, US) else cnt.get(v, 0)
        if key not in best or c > best[key][1]:
            best[key] = (v, c)
    return {v for v, c in best.values()}

def canon_country(s, vocab, vocab_by_key, vocab_by_sorted, vocab_nospace):
    if not isinstance(s, str) or not s.strip():
        return None, "missing"
    s0 = fix_mojibake(s)
    if s0 in CYR: return CYR[s0], "alias"
    if s0 in vocab: return s0, "exact"
    k = text_key(s0)
    if k in ALIASES: return ALIASES[k], "alias"
    if k in vocab_by_key: return vocab_by_key[k], "case"
    ks = " ".join(sorted(k.split()))
    if ks in vocab_by_sorted: return vocab_by_sorted[ks], "shuffled"
    if re.fullmatch(r"[a-z0-9]{2}", k) and Levenshtein.distance(k, "uk") == 1 and k != "us":
        return UK, "typo"          # 'UJ','7K','YK','UI',... : 2-char corruptions of the frequent 'UK'
    m = re.fullmatch(r"republic of (.+)", k)
    if m and m.group(1) in vocab_by_key: return vocab_by_key[m.group(1)], "alias"
    kn = k.replace(" ", "")
    if kn in vocab_nospace: return vocab_nospace[kn], "spacing"
    for a, v in ALIASES.items():
        if a.replace(" ", "") == kn: return v, "alias"
    # typo: single/double edit on no-space form
    best = process.extractOne(kn, list(vocab_nospace), scorer=Levenshtein.distance)
    if best and len(kn) >= 4 and best[1] <= (1 if len(kn) <= 7 else 2):
        return vocab_nospace[best[0]], "typo"
    # partial shuffles / drops of the long UK/US names
    if len(k.split()) >= 3:
        for full in (UK, US):
            fk = text_key(full)
            if set(k.split()) <= set(fk.split()) and len(set(k.split())) >= len(set(fk.split())) - 2:
                return full, "partial_words"
            if fuzz.token_set_ratio(k, fk) >= 90:
                return full, "partial_words"
    return s0, "unmapped"

# ---------------- genre ----------------
DISCOGS_GENRES = ["Rock", "Electronic", "Pop", "Funk / Soul", "Folk, World, & Country", "Jazz", "Hip Hop", "Reggae",
                  "Classical", "Stage & Screen", "Non-Music", "Blues", "Latin", "Children's", "Brass & Military"]
# word -> genre. Genre words are unique across genres, so any surviving word identifies its genre even
# when the injected noise dropped/shuffled words or flattened 'A|B' into 'A B'. Related sub-style words
# (punk, roll, baroque, ...) are mapped to their Discogs parent genre.
GWORD = {"rock": "Rock", "roll": "Rock", "punk": "Rock", "electronic": "Electronic", "electro": "Electronic",
         "industrial": "Electronic", "pop": "Pop", "popular": "Pop", "funk": "Funk / Soul", "soul": "Funk / Soul",
         "folk": "Folk, World, & Country", "world": "Folk, World, & Country", "country": "Folk, World, & Country",
         "jazz": "Jazz", "fusion": "Jazz", "hip": "Hip Hop", "hop": "Hip Hop", "rap": "Hip Hop", "reggae": "Reggae",
         "caribbean": "Reggae", "classical": "Classical", "baroque": "Classical", "stage": "Stage & Screen",
         "screen": "Stage & Screen", "theater": "Stage & Screen", "film": "Stage & Screen", "non": "Non-Music",
         "mus": "Non-Music", "blues": "Blues", "latin": "Latin", "children": "Children's", "kids": "Children's",
         "brass": "Brass & Military", "military": "Brass & Military"}
IGNORE_W = {"and", "s", "music", "genres", "contemporary", "art", "hard", "garage", "progressive", "classic",
            "alternative", "teen", "euro", "j", "r", "b", "digital", "noise", "roots"}
GORDER = {}  # frozenset(genres) -> most frequent clean ordering (filled in main)

def _gword(w):
    w = w.translate(str.maketrans("50149", "sorlo"))
    if w in GWORD: return GWORD[w]
    if w in IGNORE_W: return None
    if w.startswith("electro"): return ["Electronic", "Pop"] if w.endswith("pop") else "Electronic"
    if w.endswith("pop") and len(w) > 3: return "Pop"
    cands = [(Levenshtein.distance(w, k), k) for k in GWORD if abs(len(k) - len(w)) <= 1]
    lim = 1 if len(w) <= 7 else 2
    cands = sorted(c for c in cands if c[0] <= lim)
    if not cands: return None
    if len(w) <= 3 and len(cands) > 1 and cands[0][0] == cands[1][0]:
        # tie between 3-letter words (e.g. 'pip' -> pop/hip): prefer the more frequent genre
        pri = ["Pop", "Hip Hop", "Non-Music"]
        return min((GWORD[c[1]] for c in cands if c[0] == cands[0][0]), key=lambda g: pri.index(g) if g in pri else 9)
    return GWORD[cands[0][1]]

def genre_set(s):
    out = []
    for w in re.findall(r"[a-z0-9]+", s.lower().replace("r&b", "funk")):
        g = _gword(w)
        for x in (g if isinstance(g, list) else [g]):
            if x and x not in out: out.append(x)
    return out

def canon_genre(s):
    if not isinstance(s, str) or not s.strip():
        return None, "missing"
    if all(p in DISCOGS_GENRES for p in s.split("|")):
        return s, "exact"
    comp = {re.sub(r"[^a-z]", "", g.lower()): g for g in DISCOGS_GENRES}
    parts = [comp.get(re.sub(r"[^a-z]", "", p.lower())) for p in s.split("|")]
    if all(parts):   # intra-word spaces / separator noise only ('Elec tronic', 'HipHop')
        return "|".join(GORDER.get(frozenset(parts), parts)), "corrected"
    gs = genre_set(s)
    if not gs:
        return collapse_ws(s), "unmapped"
    gs = GORDER.get(frozenset(gs), gs)
    return "|".join(gs), "corrected"

def main():
    tr = {s: pd.read_pickle(f"{W}/state/translated_{s}.pkl") for s in SRCS}
    for g, c in tr["discogs"]["genre"].dropna().value_counts().items():
        parts = g.split("|")
        if all(p in DISCOGS_GENRES for p in parts) and frozenset(parts) not in GORDER:
            GORDER[frozenset(parts)] = parts   # value_counts is sorted by frequency -> most frequent order wins
    allc = pd.concat([tr["discogs"]["release-country"], tr["musicbrainz"]["release-country"]])
    vocab = build_country_vocab(allc)
    vk = {text_key(v): v for v in vocab}
    vs = {" ".join(sorted(text_key(v).split())): v for v in vocab}
    vn = {text_key(v).replace(" ", ""): v for v in vocab}
    cov = []
    frames = []
    for src, df in tr.items():
        o = pd.DataFrame({"id": df["id"], "source": src})
        for c in ["name", "artist", "release-date", "release-country", "duration", "label", "genre", "tracks"]:
            o[c + "_raw"] = df[c] if c in df else None
        o["artist"] = [clean_artist(src, a) for a in o["artist_raw"]]
        nm = [clean_name(src, n, a) for n, a in zip(o["name_raw"], o["artist_raw"])]
        o["name"] = [x[0] for x in nm]; o["name_note"] = [x[1] for x in nm]
        o["name_key"] = o["name"].map(text_key)
        o["name_dropmark"] = o["name_raw"].map(lambda x: isinstance(x, str) and "  " in x.strip())
        o["artist_key"] = o["artist"].map(lambda a: text_key(a) if isinstance(a, str) else "")
        dp = [parse_date(x) for x in o["release-date_raw"]]
        o["date"] = [x[0] for x in dp]; o["year"] = [x[1] for x in dp]; o["month"] = [x[2] for x in dp]; o["date_status"] = [x[3] for x in dp]
        cc = [canon_country(x, vocab, vk, vs, vn) for x in o["release-country_raw"]]
        o["country"] = [x[0] for x in cc]; o["country_status"] = [x[1] for x in cc]
        o["country_key"] = o["country"].map(lambda c: CKEY.get(c, c) if isinstance(c, str) else None)
        o["duration"] = [parse_duration(src, x) for x in o["duration_raw"]]
        o["tracks"] = o["tracks_raw"].map(parse_tracks)
        o["track_keys"] = o["tracks"].map(lambda t: frozenset(text_key(x) for x in t) if t else None)
        o["track_core"] = o["tracks"].map(lambda t: tuple(track_core(x) for x in t) if t else None)
        o["n_tracks"] = o["tracks"].map(lambda t: len(t) if t else None)
        o["label"] = o["label_raw"].map(lambda s: [collapse_ws(fix_mojibake(x)) for x in s.split("|") if x.strip()] if isinstance(s, str) else None)
        gg = [canon_genre(x) for x in o["genre_raw"]]
        o["genre"] = [x[0] for x in gg]; o["genre_status"] = [x[1] for x in gg]
        o.to_pickle(f"{W}/state/norm_{src}.pkl")
        frames.append(o)
        # coverage / parse diagnostics
        for attr, stat in [("release-country", "country_status"), ("genre", "genre_status"), ("release-date", "date_status")]:
            nn = o[attr + "_raw"].notna().sum()
            if nn == 0: continue
            st = o[stat].value_counts().to_dict()
            bad = st.get("unmapped", 0) + st.get("unparsed", 0) + st.get("bad_month", 0) + st.get("bad_day", 0)
            cov.append(dict(source=src, attribute=attr, non_null=int(nn), canonical=int(nn - bad), unmapped=int(bad),
                            canonical_rate=round((nn - bad) / nn, 4), detail=json.dumps({k: int(v) for k, v in st.items()})))
        for attr in ["duration", "tracks"]:
            nn = o[attr + "_raw"].notna().sum(); ok = o[attr].notna().sum()
            cov.append(dict(source=src, attribute=attr, non_null=int(nn), canonical=int(ok), unmapped=int(nn - ok),
                            canonical_rate=round(ok / max(nn, 1), 4), detail="parsed non-empty (lastfm '[]' / mb '0:00' treated as unknown)"))
    pd.DataFrame(cov).to_csv(f"{W}/taxonomy_coverage.csv", index=False)
    print(pd.DataFrame(cov).drop(columns="detail").to_string())
    allf = pd.concat(frames)
    print("unmapped countries:", allf[allf.country_status == "unmapped"].country.value_counts().to_dict())
    print("unmapped genres:", allf[allf.genre_status == "unmapped"].genre.value_counts().to_dict())
    json.dump({"genre": {"schema_taxonomy": "task/input/schemamatching/Music_Genres_Taxonomy.csv (column 'Genre Name')",
                         "decision": "Only discogs carries genre, using the 15-value Discogs top-level genre vocabulary ('Electronic','Funk / Soul',...). "
                                     "The schema's own examples ('Electronic','House','Techno') are not taxonomy Genre Names, so the taxonomy is treated as "
                                     "non-exhaustive; corrupted values are segmented/typo-corrected to the Discogs vocabulary and multi-genre values kept "
                                     "'|'-joined in source order. No mapping to 'Electronic / Dance' etc. is performed.",
                         "vocabulary": DISCOGS_GENRES, "unmapped_policy": "keep cleaned source string"},
               "release-country": {"canonical": "full English names as used by the schema examples (UK/USA -> full official names)",
                                   "aliases": ALIASES, "comparison_key_groups": CKEY,
                                   "vocab": sorted(vocab), "unmapped_policy": "keep mojibake-fixed source string"},
               "tracks": {"serialization": "JSON list in fused.csv"}, "label": {"serialization": "'|'-joined in fused.csv"}},
              open(f"{W}/taxonomy_plan.json", "w"), indent=1, ensure_ascii=False)

if __name__ == "__main__":
    main()
