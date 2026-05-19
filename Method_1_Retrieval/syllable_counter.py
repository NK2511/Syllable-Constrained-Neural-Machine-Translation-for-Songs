import nltk
from SyllableCounter_final import (
    count_english_syllables,
    count_hindi_syllables,
    _HALANT,
    _NUKTA,
    _STANDALONE_VOWELS,
    _MATRAS
)

# -----------------------------
# CMU DICTIONARY FOR ENGLISH SPLITTING
# -----------------------------
try:
    _cmu_dict = nltk.corpus.cmudict.dict()
except LookupError:
    print("CMU Dict not found. Downloading...")
    nltk.download('cmudict')
    _cmu_dict = nltk.corpus.cmudict.dict()

# -----------------------------
# DEVANAGARI CHARACTER HELPERS
# -----------------------------
def _is_devanagari_consonant(ch: str) -> bool:
    return 0x0915 <= ord(ch) <= 0x0939

def _is_devanagari_vowel(ch: str) -> bool:
    return ch in _STANDALONE_VOWELS

def _is_matra(ch: str) -> bool:
    return ch in _MATRAS
