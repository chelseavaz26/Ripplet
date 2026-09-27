"""
Twitter Demographics Profiling Module
"""

from modules.demographics.profiler import (
    INTEREST_TAXONOMY,
    GeographicResolver,
    classify_bio_interests,
    detect_bio_language,
    profile_demographics,
)

__all__ = [
    "INTEREST_TAXONOMY",
    "GeographicResolver",
    "classify_bio_interests",
    "detect_bio_language",
    "profile_demographics",
]
