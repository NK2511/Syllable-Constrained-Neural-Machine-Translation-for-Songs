"""
Syllable Splitter
=================
Splits text into individual syllable *strings* (not just counts).
Used by the GUI to render coloured syllable chips.
"""

import re
import pyphen

from syllable_counter import (
    _is_devanagari_consonant, _is_devanagari_vowel,
    _is_matra, _HALANT, _NUKTA
)

from syllable_counter import _cmu_dict   # CMU dict already loaded there

_pyphen_dic = pyphen.Pyphen(lang='en_US')


def _split_word_by_vowels(word: str, target_count: int) -> list:
    """
    Naive vowel-cluster split used when pyphen disagrees with CMU.
    Tries to cut the word into `target_count` chunks at vowel boundaries.
    e.g. "body" → ["bo", "dy"],  "gonna" → ["gon", "na"]
    """
    if target_count <= 1:
        return [word]

    # Find positions of vowel clusters
    vowel_positions = [m.start() for m in re.finditer(r'[aeiouy]+', word.lower())]
    if len(vowel_positions) >= target_count:
        # Split just before each vowel cluster (skip the first one — start of word)
        cuts = vowel_positions[1:target_count]
        parts = []
        prev = 0
        for cut in cuts:
            # Walk cut back to start of the vowel's preceding consonant cluster
            split_at = cut
            while split_at > prev + 1 and word[split_at - 1].lower() not in 'aeiouy':
                split_at -= 1
            parts.append(word[prev:split_at])
            prev = split_at
        parts.append(word[prev:])
        return [p for p in parts if p]

    # Last resort: evenly divide by character count
    n = len(word)
    size = max(1, n // target_count)
    return [word[i:i + size] for i in range(0, n, size)][:target_count]


def split_english_syllables(text: str) -> list:
    """
    Return a flat list of syllable strings for an English line.
    Uses pyphen for splitting, but cross-checks against CMU dict count.
    When they disagree, falls back to a vowel-boundary split so the
    chip count always matches the syllable number displayed.
    """
    cleaned = re.sub(r"[^a-zA-Z' ]", ' ', text)
    words = [w for w in cleaned.split() if w.strip()]
    syllables = []
    for word in words:
        # pyphen split
        hyphenated = _pyphen_dic.inserted(word)
        parts = hyphenated.split('-')

        # CMU dict count (ground truth)
        word_lower = word.lower()
        if word_lower in _cmu_dict:
            cmu_count = sum(1 for p in _cmu_dict[word_lower][0] if p[-1].isdigit())
            if cmu_count > 0 and len(parts) != cmu_count:
                parts = _split_word_by_vowels(word, cmu_count)

        syllables.extend(parts)
    return syllables



def split_hindi_syllables(text: str) -> list:
    """Return a flat list of akshar (syllable) strings for a Hindi/Devanagari line."""
    # Keep only Devanagari characters
    text = re.sub(r'[^\u0900-\u097F]', '', text)
    if not text:
        return []

    syllables = []
    current = ''
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if _is_devanagari_vowel(ch):
            if current:
                syllables.append(current)
                current = ''
            syllables.append(ch)
            i += 1

        elif _is_devanagari_consonant(ch):
            if current:
                syllables.append(current)
            current = ch
            i += 1

            # Optional nukta
            if i < n and text[i] == _NUKTA:
                current += text[i]; i += 1

            # Conjuncts: consonant + halant + consonant
            while i + 1 < n and text[i] == _HALANT and _is_devanagari_consonant(text[i + 1]):
                current += text[i] + text[i + 1]; i += 2
                if i < n and text[i] == _NUKTA:
                    current += text[i]; i += 1

            # Matra (vowel sign)
            if i < n and _is_matra(text[i]):
                current += text[i]; i += 1

            # Anusvara / visarga / chandrabindu
            while i < n and text[i] in '\u0902\u0903\u0901':
                current += text[i]; i += 1
        else:
            i += 1

    if current:
        syllables.append(current)

    return syllables
