"""
Twitter Demographics Profiling Module

Profiles user communities in time windows by analyzing bio language,
geographic locations (via pycountry), and profession/interest categories.
Strictly returns aggregated, anonymized frequency distributions.
"""

from collections import Counter
import re
from typing import Any
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import pycountry

# Ensure deterministic language detection
DetectorFactory.seed = 0

# Comprehensive Profession & Interest Keyword Taxonomy
INTEREST_TAXONOMY: dict[str, list[str]] = {
    "Healthcare & Medicine": [
        "doctor", "physician", "nurse", "surgeon", "rn", "medic", "healthcare",
        "hospital", "clinic", "pediatrician", "epidemiologist", "pharmacist",
        "therapist", "dentist", "paramedic", "virologist", "biologist", "biomedical",
        "cardiology", "oncology", "neurology", "health worker", "public health",
    ],
    "Technology & Engineering": [
        "engineer", "developer", "programmer", "coder", "software", "tech",
        "data scientist", "ai", "ml", "web dev", "sysadmin", "devops", "cloud",
        "cybersecurity", "architect", "frontend", "backend", "fullstack", "hacker",
        "python", "robotics", "computer science",
    ],
    "Business & Finance": [
        "founder", "ceo", "co-founder", "entrepreneur", "startup", "investor",
        "vc", "finance", "trader", "banker", "accounting", "consultant", "business",
        "executive", "director", "manager", "venture", "crypto", "fintech", "economist",
    ],
    "Education & Academia": [
        "student", "teacher", "professor", "educator", "academic", "researcher",
        "scholar", "phd", "scientist", "lecturer", "postdoc", "instructor",
        "faculty", "university", "college", "school", "grad student",
    ],
    "Marketing, Media & Communications": [
        "marketer", "marketing", "seo", "growth", "content creator", "journalist",
        "reporter", "editor", "writer", "author", "blogger", "podcaster", "pr",
        "communications", "media", "broadcaster", "copywriter", "columnist",
    ],
    "Creative Arts & Design": [
        "designer", "artist", "photographer", "illustrator", "musician", "filmmaker",
        "producer", "actor", "animator", "creator", "graphic design", "ui/ux",
        "creative", "poet", "author", "composer",
    ],
    "Public Service, Politics & Law": [
        "lawyer", "attorney", "advocate", "activist", "policy", "politician",
        "government", "diplomat", "public servant", "ngo", "human rights", "legal",
        "judge", "military", "veteran", "law",
    ],
    "Sports & Fitness": [
        "athlete", "coach", "fitness", "trainer", "sports", "player", "runner",
        "marathon", "bodybuilder", "cyclist", "gym", "football", "basketball", "soccer",
    ],
}

# Pre-compile regex for fast category matching
_COMPILED_PATTERNS: dict[str, re.Pattern[str]] = {
    cat: re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
    for cat, words in INTEREST_TAXONOMY.items()
}

# 2-letter US State Postal Codes
_US_STATES: set[str] = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt",
    "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri",
    "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc",
}

# High-frequency Twitter geographic aliases & abbreviations
_GEO_ALIASES: dict[str, str] = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "great britain": "United Kingdom",
    "london": "United Kingdom",
    "nyc": "United States",
    "new york": "United States",
    "los angeles": "United States",
    "california": "United States",
    "texas": "United States",
    "florida": "United States",
    "chicago": "United States",
    "washington": "United States",
    "seattle": "United States",
    "san francisco": "United States",
    "toronto": "Canada",
    "ontario": "Canada",
    "vancouver": "Canada",
    "montreal": "Canada",
    "quebec": "Canada",
    "delhi": "India",
    "new delhi": "India",
    "mumbai": "India",
    "bangalore": "India",
    "bengaluru": "India",
    "chennai": "India",
    "hyderabad": "India",
    "kolkata": "India",
    "paris": "France",
    "berlin": "Germany",
    "tokyo": "Japan",
    "sydney": "Australia",
    "melbourne": "Australia",
}


class GeographicResolver:
    """
    Resolves messy free-text location strings to canonical countries/regions
    using pycountry countries and subdivisions database with pre-indexed lookups.
    """

    def __init__(self) -> None:
        self.lookup: dict[str, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        # 1. Countries
        for country in pycountry.countries:
            canonical = country.name
            self.lookup[country.name.lower()] = canonical
            if hasattr(country, "common_name"):
                self.lookup[country.common_name.lower()] = canonical
            if hasattr(country, "official_name"):
                self.lookup[country.official_name.lower()] = canonical
            self.lookup[country.alpha_2.lower()] = canonical
            self.lookup[country.alpha_3.lower()] = canonical

        # 2. Subdivisions (states, provinces, regions)
        for sub in pycountry.subdivisions:
            c = pycountry.countries.get(alpha_2=sub.country_code)
            if c:
                sub_name = sub.name.lower()
                # Avoid collision with short common words
                if len(sub_name) >= 3:
                    self.lookup[sub_name] = c.name

        # 3. Custom Twitter geographic aliases
        for alias, canonical in _GEO_ALIASES.items():
            self.lookup[alias.lower()] = canonical

    def resolve(self, raw_location: Any) -> str:
        """
        Resolves a raw location string into a canonical country/region name.
        Returns 'unknown' if no match is found.
        """
        if raw_location is None or pd.isna(raw_location):
            return "unknown"

        if not isinstance(raw_location, str):
            return "unknown"

        loc = raw_location.strip().lower()
        if not loc:
            return "unknown"

        # Direct exact match
        if loc in self.lookup:
            return self.lookup[loc]

        # Tokenize by common location separators: comma, slash, dash, bar
        parts = [p.strip() for p in re.split(r"[,/|•\-]+", loc) if p.strip()]

        # Check rightmost component first (commonly country or state, e.g. "Seattle, WA")
        for part in reversed(parts):
            if len(part) == 2 and part in _US_STATES:
                return "United States"
            if part in self.lookup:
                return self.lookup[part]

        # Check leftmost component next (e.g. city or state name)
        for part in parts:
            if part in self.lookup:
                return self.lookup[part]

        # Substring search for significant country/region names (>= 4 chars)
        for part in parts:
            for kw, canonical in self.lookup.items():
                if len(kw) >= 4 and kw in part:
                    return canonical

        return "unknown"


# Pre-instantiate singleton resolver
_GEO_RESOLVER = GeographicResolver()


def detect_bio_language(bio: Any) -> str:
    """
    Detects language code of a user's bio using langdetect.
    Returns ISO 639-1 code (e.g. 'en', 'es') or 'unknown' for empty/unparseable text.
    """
    if bio is None or pd.isna(bio) or not isinstance(bio, str):
        return "unknown"

    bio_str = bio.strip()
    # Require at least some alphabetic characters
    if not re.search(r"[a-zA-Z\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF]", bio_str):
        return "unknown"

    try:
        return detect(bio_str)
    except LangDetectException:
        return "unknown"


def classify_bio_interests(bio: Any) -> list[str]:
    """
    Matches bio text against domain taxonomy to identify interest/profession categories.
    Returns list of matched categories, or ['Other / Unclassified'] if none match.
    """
    if bio is None or pd.isna(bio) or not isinstance(bio, str):
        return ["Other / Unclassified"]

    bio_str = bio.strip()
    if not bio_str:
        return ["Other / Unclassified"]

    matched = [cat for cat, pat in _COMPILED_PATTERNS.items() if pat.search(bio_str)]
    if not matched:
        return ["Other / Unclassified"]
    return matched


def profile_demographics(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Profiles user demographics for a time window DataFrame of tweets.

    Deduplicates tweets to unique users, detects bio language, resolves
    geographic distribution via pycountry, and classifies professions/interests.

    Strict Privacy Guarantee:
        Returns ONLY aggregated counts. Never exposes per-user records,
        user IDs, tweet IDs, or individual identifiable profiles.

    Args:
        df: pd.DataFrame with tweet records from pipeline.get_window().

    Returns:
        dict with structure:
            {
                "languages": {"en": 120, "es": 15, ...},
                "regions": {"United States": 45, "United Kingdom": 20, "unknown": 10, ...},
                "interest_categories": {"Healthcare & Medicine": 30, ...}
            }
    """
    if df is None or df.empty:
        return {
            "languages": {},
            "regions": {},
            "interest_categories": {},
        }

    # 1. Deduplicate to unique users
    if "user_id" in df.columns and df["user_id"].notna().any():
        # Deduplicate by user_id
        unique_users = df.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"])
    elif "bio" in df.columns and "location" in df.columns:
        # Fallback deduplication by (bio, location)
        unique_users = df.drop_duplicates(subset=["bio", "location"])
    elif "bio" in df.columns:
        unique_users = df.drop_duplicates(subset=["bio"])
    else:
        unique_users = df.drop_duplicates()

    languages_counter: Counter[str] = Counter()
    regions_counter: Counter[str] = Counter()
    interests_counter: Counter[str] = Counter()

    # 2. Extract and aggregate per-user traits
    bios = unique_users["bio"] if "bio" in unique_users.columns else pd.Series([None] * len(unique_users))
    locations = unique_users["location"] if "location" in unique_users.columns else pd.Series([None] * len(unique_users))

    for bio, loc in zip(bios, locations):
        # A. Language detection
        lang = detect_bio_language(bio)
        languages_counter[lang] += 1

        # B. Geographic region matching
        region = _GEO_RESOLVER.resolve(loc)
        regions_counter[region] += 1

        # C. Interest / profession classification
        cats = classify_bio_interests(bio)
        for cat in cats:
            interests_counter[cat] += 1

    # 3. Format into strictly aggregated distributions sorted descending by frequency
    return {
        "languages": dict(languages_counter.most_common()),
        "regions": dict(regions_counter.most_common()),
        "interest_categories": dict(interests_counter.most_common()),
    }
