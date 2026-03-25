"""
Syllable Counter Module
=======================
Counts syllables for both English and Hindi text.

English: Uses CMU Pronouncing Dictionary (phoneme-based) with pyphen fallback.
Hindi:   Uses akshar (orthographic syllable) counting based on Unicode analysis.
"""

import re
import pyphen
import nltk

# Download CMU dict if not present
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', quiet=True)

from nltk.corpus import cmudict

# ─── English Syllable Counter ───────────────────────────────────────────────

_cmu_dict = cmudict.dict()
_pyphen_dic = pyphen.Pyphen(lang='en_US')


def _count_syllables_cmu(word: str) -> int | None:
    """Count syllables using CMU Pronouncing Dictionary (phoneme-based)."""
    word_lower = word.lower().strip()
    if word_lower in _cmu_dict:
        # Count vowel phonemes (digits in CMU phonemes indicate stress on vowels)
        phonemes = _cmu_dict[word_lower][0]
        return sum(1 for p in phonemes if p[-1].isdigit())
    return None


def _count_syllables_pyphen(word: str) -> int:
    """Count syllables using hyphenation (fallback)."""
    hyphenated = _pyphen_dic.inserted(word)
    return max(1, hyphenated.count('-') + 1)


def _count_syllables_heuristic(word: str) -> int:
    """Last-resort heuristic syllable counter for unknown English words."""
    word = word.lower().strip()
    if not word:
        return 0
    
    # Count vowel groups
    count = len(re.findall(r'[aeiouy]+', word))
    
    # Adjust for silent-e
    if word.endswith('e') and not word.endswith('le'):
        count -= 1
    
    # Adjust for -ed endings (usually silent)
    if word.endswith('ed') and len(word) > 3 and word[-3] not in 'dt':
        count -= 1
    
    return max(1, count)


def count_english_syllables(text: str) -> int:
    """
    Count total syllables in an English text line.
    Uses CMU dict → pyphen → heuristic fallback chain.
    """
    # Clean: keep only letters, apostrophes, spaces
    cleaned = re.sub(r"[^a-zA-Z' ]", ' ', text)
    words = [w for w in cleaned.split() if w.strip()]
    
    total = 0
    for word in words:
        # Try CMU dict first (most accurate for standard words)
        cmu_count = _count_syllables_cmu(word)
        if cmu_count is not None:
            total += cmu_count
        else:
            # Try pyphen
            pyphen_count = _count_syllables_pyphen(word)
            if pyphen_count > 0:
                total += pyphen_count
            else:
                total += _count_syllables_heuristic(word)
    
    return total


# ─── Hindi Syllable (Akshar) Counter ────────────────────────────────────────

# Unicode ranges for Devanagari
_DEVANAGARI_VOWELS = set(
    'अआइईउऊऋॠऌॡएऐओऔ'
)

# Vowel signs (matras) - these modify consonants but don't create new syllables
_DEVANAGARI_MATRAS = set(
    'ा ि ी ु ू ृ ॄ ॢ ॣ े ै ो ौ'.split()
)
# Also as individual chars
_MATRA_CODEPOINTS = {
    '\u093E',  # ा
    '\u093F',  # ि
    '\u0940',  # ी
    '\u0941',  # ु
    '\u0942',  # ू
    '\u0943',  # ृ
    '\u0944',  # ॄ
    '\u0962',  # ॢ
    '\u0963',  # ॣ
    '\u0947',  # े
    '\u0948',  # ै
    '\u094B',  # ो
    '\u094C',  # ौ
}

_HALANT = '\u094D'  # ्  (virama - joins consonants)
_NUKTA = '\u093C'   # ़

# Devanagari consonant range: 0x0915 - 0x0939
def _is_devanagari_consonant(ch: str) -> bool:
    cp = ord(ch)
    return 0x0915 <= cp <= 0x0939 or ch == 'ड़' or ch == 'ढ़'

def _is_devanagari_vowel(ch: str) -> bool:
    return ch in _DEVANAGARI_VOWELS

def _is_matra(ch: str) -> bool:
    return ch in _MATRA_CODEPOINTS


def count_hindi_syllables(text: str) -> int:
    """
    Count syllables (akshar) in Hindi/Devanagari text.
    
    Rules:
    - Each standalone vowel = 1 akshar
    - Consonant + (optional matra) = 1 akshar
    - Consonant + halant + consonant = conjunct (counts as 1 akshar for the group)
    - Anusvara (ं) and visarga (ः) don't create new syllables
    """
    # Remove all non-Devanagari characters
    text = re.sub(r'[^\u0900-\u097F]', '', text)
    
    if not text:
        return 0
    
    syllable_count = 0
    i = 0
    n = len(text)
    
    while i < n:
        ch = text[i]
        
        if _is_devanagari_vowel(ch):
            # Standalone vowel = 1 syllable
            syllable_count += 1
            i += 1
            
        elif _is_devanagari_consonant(ch):
            # Start of a syllable
            syllable_count += 1
            i += 1
            
            # Skip nukta if present
            if i < n and text[i] == _NUKTA:
                i += 1
            
            # Handle conjuncts: consonant + halant + consonant(s)
            while i + 1 < n and text[i] == _HALANT and _is_devanagari_consonant(text[i + 1]):
                i += 2  # skip halant + next consonant
                # Skip nukta after conjunct consonant
                if i < n and text[i] == _NUKTA:
                    i += 1
            
            # Skip matra (vowel sign) - part of this syllable
            if i < n and _is_matra(text[i]):
                i += 1
            
            # Skip anusvara (ं), visarga (ः), chandrabindu (ँ)
            while i < n and text[i] in '\u0902\u0903\u0901':
                i += 1
        else:
            # Other Devanagari marks (skip)
            i += 1
    
    return syllable_count


# ─── Unified Interface ──────────────────────────────────────────────────────

def count_syllables(text: str, language: str = 'en') -> int:
    """
    Count syllables in text.
    
    Args:
        text: Input text
        language: 'en' for English, 'hi' for Hindi
    
    Returns:
        Total syllable count
    """
    if language == 'en':
        return count_english_syllables(text)
    elif language == 'hi':
        return count_hindi_syllables(text)
    else:
        raise ValueError(f"Unsupported language: {language}. Use 'en' or 'hi'.")


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # English tests
    test_en = [
        "I'm walking on sunshine",
        "Hello is it me you're looking for",
        "We will rock you",
        "Bohemian Rhapsody",
        "Shape of you",
    ]
    
    print("=== English Syllable Counts ===")
    for line in test_en:
        count = count_english_syllables(line)
        print(f"  [{count:2d}] {line}")
    
    # Hindi tests
    test_hi = [
        "तुम ही हो",
        "दिल में तेरी",
        "ज़िंदगी गुलज़ार है",
        "तेरे बिना ज़िंदगी से",
        "इश्क़ वाला लव",
    ]
    
    print("\n=== Hindi Syllable (Akshar) Counts ===")
    for line in test_hi:
        count = count_hindi_syllables(line)
        print(f"  [{count:2d}] {line}")
