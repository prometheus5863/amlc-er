"""Name and address normalisation, country-aware but open-set.

Everything here is rule-based and uses only the provided data plus generic
language knowledge (legal-form words, street abbreviations, script
transliteration). No external lookups.

Main entry point: `normalize_frame(df)` adds these columns to a source table:

  name_clean     lower-case, accent-folded, noise-stripped full name
  name_core      name without legal forms / honorifics (space-joined tokens)
  name_concat    name_core with spaces removed ("yeagerstaking")
  name_skel      consonant skeleton of name_concat (robust to vowel/translit noise)
  legal          canonical legal-form class ("llc", "inc", "pvt_ltd", "sarl", ...)
  is_domain      name looked like a website ("s6first.com", "@yeagerstaking")
  is_translit    name was in Devanagari and was transliterated
  addr_clean     normalised address (abbreviations expanded, junk removed)
  house_no       first number in the address (leading zeros stripped) or ""
  addr_nums      space-joined set of all numbers in the address
  street         first street-like component after the house number
  postcode       5-digit (FR/US zip) or 6-digit (IN pin) code if present
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

# --------------------------------------------------------------------------- #
# character level
# --------------------------------------------------------------------------- #
_LATIN_FOLD = {}
for cp in range(0xC0, 0x250):
    ch = chr(cp)
    base = unicodedata.normalize("NFKD", ch)[0]
    if base.isascii() and base.isalpha():
        _LATIN_FOLD[cp] = base
_LATIN_FOLD.update({ord("ß"): "ss", ord("æ"): "ae", ord("œ"): "oe", ord("Æ"): "AE", ord("Œ"): "OE",
                    ord("’"): "'", ord("‘"): "'", ord("“"): '"', ord("”"): '"', ord("–"): "-", ord("—"): "-"})

# Devanagari -> Latin (simplified ITRANS-like). Inherent 'a' is dropped at the
# end of words; matras replace it. Good enough for fuzzy / skeleton matching.
_DEV_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l", "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh",
    "फ़": "f", "य़": "y",
}
_DEV_VOW = {"ऎ": "e", "ऒ": "o", "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai",
            "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_DEV_MATRA = {"ॆ": "e", "ॊ": "o", "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
              "ौ": "au", "ॉ": "o", "ॅ": "e"}
_DEV_SIGN = {"ं": "n", "ँ": "n", "ः": "h", "्": "", "़": ""}
_DEV_DIGITS = {chr(0x966 + i): str(i) for i in range(10)}
# frequent business words: transliteration alone gets these badly wrong
_DEV_WORDS = {
    "प्राइवेट": "private", "प्रा.": "pvt", "प्रा": "pvt", "लिमिटेड": "limited", "लि.": "ltd", "लि": "ltd",
    "एलएलपी": "llp", "इंटरनेशनल": "international", "इन्टरनेशनल": "international", "टेक्नोलॉजीज": "technologies",
    "टेक्नोलॉजी": "technology", "इंडिया": "india", "इण्डिया": "india", "सर्विसेज": "services", "सर्विसेस": "services",
    "सर्विस": "service", "इंडस्ट्रीज": "industries", "एंटरप्राइजेज": "enterprises", "एंटरप्राइज": "enterprise",
    "सॉल्यूशंस": "solutions", "सोल्यूशंस": "solutions", "कंसल्टेंसी": "consultancy", "कंसल्टेंट्स": "consultants",
    "मार्केटिंग": "marketing", "ट्रेडर्स": "traders", "ट्रेडिंग": "trading", "प्रॉपर्टीज": "properties",
    "इंफ्रा": "infra", "इन्फ्रा": "infra", "एग्रो": "agro", "फूड्स": "foods", "फूड": "food", "सिस्टम्स": "systems",
    "ग्रुप": "group", "होटल": "hotel", "हेल्थकेयर": "healthcare", "फार्मा": "pharma", "मोटर्स": "motors",
    "कंस्ट्रक्शन": "construction", "कंस्ट्रक्शंस": "constructions", "इंजीनियरिंग": "engineering",
    "इंजीनियर्स": "engineers", "एंड": "and", "न्यू": "new", "भारत": "bharat", "श्री": "shri", "राम": "ram",
    "शिव": "shiv", "शिवा": "shiva", "बेकरी": "bakery", "पावर": "power", "यूनिक": "unique", "यूनाइटेड": "united",
    "प्रोड्यूसर": "producer", "वेंचर्स": "ventures", "ऑल": "all", "कॉर्पोरेशन": "corporation", "कंपनी": "company",
    "लॉजिस्टिक्स": "logistics", "एक्सपोर्ट्स": "exports", "इम्पेक्स": "impex", "इंपेक्स": "impex",
    "डेवलपर्स": "developers", "बिल्डर्स": "builders", "एसोसिएट्स": "associates", "रियल्टी": "realty",
    "एस्टेट": "estate", "मेडिकल": "medical", "ऑटो": "auto", "ऑटोमोबाइल्स": "automobiles", "स्टील": "steel",
    "केमिकल्स": "chemicals", "टेक": "tech", "इलेक्ट्रिकल्स": "electricals", "इलेक्ट्रॉनिक्स": "electronics",
    "टेक्सटाइल्स": "textiles", "फैशन": "fashion", "मीडिया": "media", "डिजिटल": "digital", "ग्लोबल": "global",
    "नेटवर्क": "network", "सॉफ्टवेयर": "software", "कैपिटल": "capital", "फाइनेंस": "finance", "एजुकेशन": "education",
}
_DEV_RE = re.compile(r"[ऀ-ॿ]")
_NONLATIN_RE = re.compile(r"[ऀ-෿฀-࿿]")  # Indic scripts (Devanagari..Sinhala etc.)


def _translit_word(w: str) -> str:
    if w in _DEV_WORDS:
        return _DEV_WORDS[w]
    w = unicodedata.normalize("NFD", w).replace("़", "")  # drop nukta
    out, i, n = [], 0, len(w)
    while i < n:
        ch = w[i]
        if ch in _DEV_CONS:
            out.append(_DEV_CONS[ch])
            nxt = w[i + 1] if i + 1 < n else ""
            if nxt in _DEV_MATRA:
                out.append(_DEV_MATRA[nxt]); i += 2; continue
            if nxt == "्":  # halant: no inherent vowel
                i += 2; continue
            if nxt and (nxt in _DEV_CONS or nxt in ("ं", "ँ")):
                out.append("a")
            i += 1
        elif ch in _DEV_VOW:
            out.append(_DEV_VOW[ch]); i += 1
        elif ch in _DEV_SIGN:
            out.append(_DEV_SIGN[ch]); i += 1
        elif ch in _DEV_MATRA:
            out.append(_DEV_MATRA[ch]); i += 1
        elif ch in _DEV_DIGITS:
            out.append(_DEV_DIGITS[ch]); i += 1
        else:
            out.append(ch); i += 1
    return "".join(out)


# Indic scripts share one layout: the same offset inside each 128-char block is
# the same letter. Map Bengali/Gurmukhi/Gujarati/Oriya/Tamil/Telugu/Kannada/
# Malayalam onto Devanagari, then use one transliterator.
_INDIC_BLOCKS = (0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_TO_DEV = {cp: chr(cp - base + 0x0900) for base in _INDIC_BLOCKS for cp in range(base, base + 0x80)}
_INDIC_RE = re.compile(r"[\u0900-\u0D7F]")


def transliterate(s: str) -> str:
    if not _INDIC_RE.search(s):
        return s
    s = s.translate(_TO_DEV)
    out = " ".join(_fix_translit_legal(_translit_word(w)) for w in s.split())
    return re.sub(r"\blimit t\b", "limited", out)  # Tamil-script "limited" comes out as "limit t"


_TR_LEGAL = [(re.compile(r"^p[iy]?r[ae]?[iy]?[vbw]h?[ae]?t[ae]?$"), "private"), (re.compile(r"^pa[iy]r[ae]?t[ae]?$"), "private"), (re.compile(r"^l[iy]m[iy]?t[ae]?[dt]$"), "limited"),
             (re.compile(r"^t?[ae]?l[ae]?m[ae]?t[ae]?d$"), "limited"), (re.compile(r"^pr[ae]?a?$"), "pvt")]


def _fix_translit_legal(w: str) -> str:
    for rx, rep in _TR_LEGAL:
        if rx.match(w):
            return rep
    return w


def fold(s: str) -> str:
    """NFKC + Latin accent folding + lower-case. Non-Latin scripts kept."""
    s = unicodedata.normalize("NFKC", s)
    return s.translate(_LATIN_FOLD).lower()


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
# multi-word legal forms first (matched on the token stream after punctuation removal)
_LEGAL_PHRASES = [
    ("private limited", "pvt_ltd"), ("pvt ltd", "pvt_ltd"), ("pvt limited", "pvt_ltd"), ("private ltd", "pvt_ltd"),
    ("p ltd", "pvt_ltd"), ("opc private limited", "pvt_ltd"), ("public limited", "ltd"),
    ("limited liability partnership", "llp"), ("limited liability company", "llc"),
    ("l l c", "llc"), ("l l p", "llp"), ("s a s u", "sasu"), ("s a s", "sas"), ("s a r l", "sarl"),
    ("s a", "sa"), ("e u r l", "eurl"), ("et fils", "fils"), ("et cie", "cie"),
]
_LEGAL_WORDS = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "ltd": "ltd", "limited": "ltd", "llp": "llp", "lp": "lp", "pllc": "llc", "plc": "ltd",
    "pc": "pc", "pvt": "pvt_ltd", "private": "pvt_ltd", "opc": "pvt_ltd",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "scop": "scop", "selarl": "selarl", "gie": "gie", "cie": "cie", "fils": "fils", "associes": "assoc",
    "associates": "assoc", "gmbh": "gmbh",
}
_HONORIFICS = {"the", "mr", "mrs", "ms", "miss", "dr", "shri", "sri", "smt", "shree", "m/s", "ms.", "messrs", "la",
               "le", "les", "l", "de", "du", "des", "et", "and", "of"}
_ALIAS_RE = re.compile(r"\b(?:f/k/a|fka|d/b/a|dba|a/k/a|aka|formerly|trading as|t/a)\b")
_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|in|co\.in|co|fr|io|biz|info|us)\b")
_JUNK_EDGE = re.compile(r"^[\W_]+|[\W_]+$")
_PUNCT = re.compile(r"[^\w\s]|_")
_WS = re.compile(r"\s+")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "7": "t", "8": "b"})
_HAS_ALPHA = re.compile(r"[a-z]")


def _deleet(tok: str) -> str:
    """'s0il' -> 'soil', 'onco1ogy' -> 'oncology'; leaves pure numbers ('84', '3033') alone."""
    if tok.isdigit() or not _HAS_ALPHA.search(tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    if letters >= 2 and any(c.isdigit() for c in tok):
        return tok.translate(_LEET)
    return tok


@lru_cache(maxsize=1 << 20)
def norm_name(raw: str):
    """-> (name_clean, name_core, legal, is_domain, is_translit, alias_core)"""
    s = raw or ""
    is_tr = bool(_INDIC_RE.search(s))
    s = fold(transliterate(s)) if is_tr else fold(s)
    s = s.replace("null", " ").strip()
    is_dom = s.lstrip(" -<>#*").startswith("@")
    s = _JUNK_EDGE.sub("", s)
    m = _DOMAIN_RE.match(s.lstrip("@"))
    if m:
        is_dom, s = True, m.group(1).replace("-", " ")
    elif is_dom and " " not in s:
        pass
    alias = ""
    parts = _ALIAS_RE.split(s)
    if len(parts) > 1:
        s, alias = parts[0], " ".join(parts[1:])
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s.replace("m/s", " ").replace("'", ""))
    toks = _join_initials([_deleet(t) for t in _WS.sub(" ", s).strip().split()])
    clean = " ".join(toks)
    core, legal = _strip_legal(toks)
    alias_core = ""
    if alias:
        at = [_deleet(t) for t in _WS.sub(" ", _PUNCT.sub(" ", alias.replace("&", " and "))).split()]
        alias_core = " ".join(_strip_legal(at)[0])
    return clean, " ".join(core), legal, is_dom, is_tr, alias_core


def _join_initials(toks):
    """'p c' -> 'pc', 'm d' -> 'md', 'l l c' -> 'llc' (runs of single letters)."""
    out, run = [], []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if t:
            out.append(t)
    return out


def _strip_legal(toks):
    s = " " + " ".join(toks) + " "
    legal = []
    for ph, cls in _LEGAL_PHRASES:
        if f" {ph} " in s:
            s = s.replace(f" {ph} ", " ")
            legal.append(cls)
    out = []
    for t in s.split():
        if t in _LEGAL_WORDS:
            legal.append(_LEGAL_WORDS[t])
        elif t in _HONORIFICS:
            continue
        else:
            out.append(t)
    if not out:  # name was only legal words / honorifics: keep the tokens
        out = [t for t in toks if t not in _HONORIFICS] or toks
    legal = sorted(set(legal))
    if "pvt_ltd" in legal:
        legal = [x for x in legal if x != "ltd"]
    return out, "|".join(legal)


_VOWELS = str.maketrans("", "", "aeiouy")


def skeleton(s: str) -> str:
    """Consonant skeleton with runs collapsed: 'intaranesanala' ~ 'international'."""
    s = s.translate(_VOWELS).replace("h", "")
    return re.sub(r"(.)\1+", r"\1", s)


# --------------------------------------------------------------------------- #
# addresses
# --------------------------------------------------------------------------- #
_ABBR_COMMON = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court",
    "crt": "court", "pl": "place", "pkwy": "parkway", "pky": "parkway", "hwy": "highway", "cir": "circle",
    "trl": "trail", "tr": "trail", "ter": "terrace", "terr": "terrace", "sq": "square", "mt": "mount",
    "ft": "fort", "hts": "heights", "xing": "crossing", "expy": "expressway", "fwy": "freeway", "tpke": "turnpike",
    "cv": "cove", "pt": "point", "rdg": "ridge", "run": "run", "way": "way", "aly": "alley", "loop": "loop",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
    "bldg": "building", "no": "number", "nr": "near", "opp": "opposite", "vill": "village", "po": "post",
    "dist": "district", "distt": "district", "ngr": "nagar", "mkt": "market", "sec": "sector", "ph": "phase",
    "h": "house", "hno": "house", "blk": "block", "extn": "extension", "ext": "extension", "cplx": "complex",
}
_ABBR_FR = {  # applied only when country == France (st -> saint, not street)
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard", "imp": "impasse",
    "pl": "place", "fg": "faubourg", "fbg": "faubourg", "rte": "route", "che": "chemin", "ch": "chemin",
    "chem": "chemin", "all": "allee", "sq": "square", "qu": "quai", "st": "saint", "ste": "sainte",
    "crs": "cours", "pass": "passage", "res": "residence", "lot": "lotissement", "zi": "zone industrielle",
    "za": "zone artisanale", "bis": "bis", "ter": "ter",
}
_ADDR_JUNK = {"null", "n/a", "na", "none", "nan", "unknown", "pmb", "#"}
_NUM_RE = re.compile(r"\d+")
_POST_RE = re.compile(r"(?<!\d)(\d{6}|\d{5}|\d{3}\s\d{3})(?!\d)")


@lru_cache(maxsize=1 << 20)
def norm_address(raw: str, country: str = ""):
    """-> (addr_clean, house_no, nums, street, postcode)"""
    s = fold(raw or "")
    s = re.sub(r"(?<![a-z])n/a(?![a-z])", " ", s)
    s = _NONLATIN_RE.sub(" ", s)  # native-script state names are handled by learned aliases later
    fr = country.lower() == "france"
    abbr = {**_ABBR_COMMON, **_ABBR_FR} if fr else _ABBR_COMMON
    comps = []
    for comp in s.split(","):
        comp = comp.replace("#", " ").replace("'", " ")
        comp = _PUNCT.sub(" ", comp.replace("/", " / ")).replace("/", " ")
        toks = []
        for t in comp.split():
            if t in _ADDR_JUNK:
                continue
            t = abbr.get(t, t)
            if t.isdigit():
                t = t.lstrip("0") or "0"
            toks.append(t)
        if toks:
            comps.append(" ".join(toks))
    clean = ", ".join(comps)
    nums = _NUM_RE.findall(clean)
    post = ""
    pm = _POST_RE.search(fold(raw or ""))
    if pm:
        post = pm.group(1).replace(" ", "")
    house = ""
    street = ""
    for comp in comps:
        m = re.match(r"^(\d+)\s+([a-z].*)$", comp)
        if m:
            house, street = m.group(1), m.group(2)
            break
    if not house and nums:
        house = nums[0]
    if not street and comps:
        street = max(comps, key=lambda c: sum(ch.isalpha() for ch in c))
    nums_s = " ".join(sorted(set(n for n in nums if n != post)))
    return clean, house, nums_s, street, post


# --------------------------------------------------------------------------- #
# frame-level
# --------------------------------------------------------------------------- #
def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Adds normalised columns to a source table (entity_id, business_name, business_address, country)."""
    out = df.copy()
    nm = [norm_name(x) for x in out.business_name.to_numpy()]
    out["name_clean"] = [x[0] for x in nm]
    out["name_core"] = [x[1] for x in nm]
    out["legal"] = [x[2] for x in nm]
    out["is_domain"] = [x[3] for x in nm]
    out["is_translit"] = [x[4] for x in nm]
    out["alias_core"] = [x[5] for x in nm]
    out["name_concat"] = out.name_core.str.replace(" ", "", regex=False)
    out["name_skel"] = [skeleton(x) for x in out.name_concat.to_numpy()]
    ad = [norm_address(a, c) for a, c in zip(out.business_address.to_numpy(), out.country.to_numpy())]
    out["addr_clean"] = [x[0] for x in ad]
    out["house_no"] = [x[1] for x in ad]
    out["addr_nums"] = [x[2] for x in ad]
    out["street"] = [x[3] for x in ad]
    out["postcode"] = [x[4] for x in ad]
    out["src"] = out.entity_id.str[:2]
    norm_name.cache_clear()
    norm_address.cache_clear()
    return out
