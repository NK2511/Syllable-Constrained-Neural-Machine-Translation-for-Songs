"""
syllable_pipeline_demo.py
=========================
Demonstrates English and Hindi syllable counting pipelines.
Shows every intermediate step for 20 sentences each.
No hardcoded expected values — output is computed entirely from the pipeline.

English pipeline:
    text → espeak-ng (phonemizer) → raw IPA
         → collapse diphthongs → count vowel nuclei

Hindi pipeline:
    Devanagari → parse into units → Ohala (1983) schwa deletion
              → Pandey (1990) cluster validity check → count nuclei

References:
    Ohala (1983). Aspects of Hindi Phonology.
    Pandey (1990). Hindi schwa deletion. Lingua, 82, 277-311.

Install:
    pip install phonemizer
    sudo apt install espeak-ng
"""
import os
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PATH"] = r"C:\Program Files\eSpeak NG" + os.pathsep + os.environ["PATH"]

import re
from typing import List, Dict, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# ENGLISH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

_DIPHTHONGS_EN: List[str] = sorted([
    'eɪ', 'aɪ', 'ɔɪ',
    'aʊ', 'əʊ', 'oʊ',
    'ɪə', 'eə', 'ʊə',
], key=len, reverse=True)

_DIPHTHONG_TOKEN = '\uE000'

_MONOPHTHONGS_EN = frozenset({
    'æ', 'ɑ', 'ɒ', 'ʌ', 'ɛ', 'e', 'ɜ', 'ə',
    'ɐ',        # near-open central: espeak-ng emits this for unstressed "a"
    'ɪ', 'i', 'ʊ', 'u', 'o', 'ɔ', 'ɚ', 'ɝ',
})


def english_steps(text: str) -> Tuple[str, str, List[str], int]:
    """
    Run English pipeline, returning intermediate steps.
    Returns: (ipa_raw, ipa_collapsed, nuclei_list, syllable_count)
    """
    from phonemizer import phonemize

    ipa_raw = phonemize(
        text,
        language='en-us',
        backend='espeak',
        with_stress=True,
        preserve_punctuation=False,
        njobs=1,
    ).strip()

    ipa_work = ipa_raw
    found_diph = [d for d in _DIPHTHONGS_EN if d in ipa_work]
    for d in _DIPHTHONGS_EN:
        ipa_work = ipa_work.replace(d, _DIPHTHONG_TOKEN)

    nuclei = []
    for ch in ipa_work:
        if ch == _DIPHTHONG_TOKEN:
            nuclei.append('[diph]')
        elif ch in _MONOPHTHONGS_EN:
            nuclei.append(ch)

    return ipa_raw, ipa_work, nuclei, len(nuclei)


# ══════════════════════════════════════════════════════════════════════════════
# HINDI PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

_HALANT       = '\u094D'
_ANUSVARA     = '\u0902'
_CHANDRABINDU = '\u0901'
_NUKTA        = '\u093C'

_STANDALONE_VOWELS = frozenset({
    '\u0905', '\u0906', '\u0907', '\u0908', '\u0909', '\u090A',
    '\u090B', '\u0960', '\u090C', '\u0961', '\u090F', '\u0910',
    '\u0913', '\u0914',
})

_MATRAS = frozenset({
    '\u093E', '\u093F', '\u0940', '\u0941', '\u0942', '\u0943',
    '\u0944', '\u0947', '\u0948', '\u094B', '\u094C', '\u0962', '\u0963',
})

_SEMIVOWELS = frozenset({'\u092F', '\u0935'})  # य व

def _is_consonant(ch: str) -> bool:
    return 0x0915 <= ord(ch) <= 0x0939

_VALID_ONSETS = frozenset({
    ('\u092A','\u0930'), ('\u092C','\u0930'), ('\u0924','\u0930'),
    ('\u0926','\u0930'), ('\u0915','\u0930'), ('\u0916','\u0930'),
    ('\u0917','\u0930'), ('\u0936','\u0930'), ('\u0938','\u0930'),
    ('\u0939','\u0930'), ('\u0927','\u0930'), ('\u0925','\u0930'),
    ('\u092D','\u0930'), ('\u092B','\u0930'), ('\u0935','\u0930'),
    ('\u0921','\u0930'),
    ('\u0924','\u0935'), ('\u0926','\u0935'), ('\u0915','\u0935'),
    ('\u0938','\u0935'), ('\u0936','\u0935'),
    ('\u092A','\u092F'), ('\u092C','\u092F'), ('\u0924','\u092F'),
    ('\u0928','\u092F'), ('\u0915','\u092F'), ('\u0917','\u092F'),
    ('\u0938','\u092F'), ('\u0936','\u092F'), ('\u0939','\u092F'),
    ('\u0927','\u092F'), ('\u092D','\u092F'), ('\u092E','\u092F'),
    ('\u0935','\u092F'),
    ('\u092A','\u0932'), ('\u092C','\u0932'), ('\u0915','\u0932'),
    ('\u0917','\u0932'), ('\u0938','\u0932'),
    ('\u0938','\u0924'), ('\u0938','\u0925'), ('\u0938','\u0928'),
    ('\u0938','\u092A'), ('\u0938','\u0915'), ('\u0938','\u0916'),
    ('\u0936','\u0928'), ('\u0936','\u0915'),
})


def _parse(word: str) -> List[Dict]:
    """Parse Devanagari word into phonological units."""
    chars = list(re.sub(r'[^\u0900-\u097F]', '', word))
    n = len(chars); units: List[Dict] = []; i = 0
    while i < n:
        ch = chars[i]
        if ch in _STANDALONE_VOWELS:
            u = {'base':ch,'type':'vowel','matra':ch,'halant':False,'anusvara':False}
            i += 1
            if i < n and chars[i] in (_ANUSVARA, _CHANDRABINDU):
                u['anusvara'] = True; i += 1
            units.append(u)
        elif _is_consonant(ch):
            u = {'base':ch,'type':'consonant','matra':None,'halant':False,'anusvara':False}
            i += 1
            if i < n and chars[i] == _NUKTA: i += 1
            if i < n and chars[i] == _HALANT:
                u['halant'] = True; i += 1
            elif i < n and chars[i] in _MATRAS:
                u['matra'] = chars[i]; i += 1
            if i < n and chars[i] in (_ANUSVARA, _CHANDRABINDU):
                u['anusvara'] = True; i += 1
            units.append(u)
        else:
            i += 1
    return units


def _next_vb(units: List[Dict], i: int) -> int:
    j = i + 1
    while j < len(units):
        u = units[j]
        if u['type'] == 'vowel': return j
        if u['type'] == 'consonant' and not u['halant']: return j
        j += 1
    return -1


def _halant_between(units: List[Dict], i: int, j: int) -> bool:
    return any(units[k]['halant'] for k in range(i + 1, j))


def hindi_word_steps(word: str) -> Tuple[List[Dict], List[str], int]:
    """
    Run Hindi pipeline on one word, returning intermediate steps.
    Returns: (units, decisions_per_unit, syllable_count)
    """
    units = _parse(word)
    n = len(units)
    if n == 0:
        return units, [], 0

    last_nc = -1
    for k in range(n - 1, -1, -1):
        u = units[k]
        if u['type'] == 'vowel' or (u['type'] == 'consonant' and not u['halant']):
            last_nc = k; break

    syllables = 0
    lv = -1
    decisions = []

    for i, u in enumerate(units):
        ch = u['base']

        if u['type'] == 'vowel' or u['matra'] is not None:
            syllables += 1; lv = i
            tag = 'matra' if u['matra'] and u['type']=='consonant' else 'vowel'
            decisions.append(f"{ch}→+1({tag})")
            continue

        if u['halant']:
            decisions.append(f"{ch}→0(halant,no vowel)")
            continue

        # inherent schwa — run rules
        if i == last_nc and u['base'] in _SEMIVOWELS:
            syllables += 1; lv = i
            decisions.append(f"{ch}→+1(semivowel,word-final retained)")
            continue

        if i == last_nc:
            decisions.append(f"{ch}→0(word-final,schwa deleted)")
            continue

        ri = _next_vb(units, i)
        if ri == -1:
            syllables += 1; lv = i
            decisions.append(f"{ch}→+1(no right context,schwa survives)")
            continue

        ru = units[ri]

        if _halant_between(units, i, ri):
            syllables += 1; lv = i
            decisions.append(f"{ch}→+1(right=CCV not CV,Ohala off)")
            continue

        right_cv = (
            ru['type'] == 'vowel' or ru['matra'] is not None
            or (ru['type'] == 'consonant' and not ru['halant'] and ri != last_nc)
            or (ru['type'] == 'consonant' and ri == last_nc and ru['base'] in _SEMIVOWELS)
        )

        if not right_cv:
            syllables += 1; lv = i
            decisions.append(f"{ch}→+1(right not CV,schwa survives)")
            continue

        cb = sum(1 for k in range(lv + 1, i) if units[k]['type'] == 'consonant')
        lha = (lv >= 0 and units[lv]['anusvara'])
        if lha: cb += 1

        if not (lv >= 0 and cb <= 1):
            syllables += 1; lv = i
            decisions.append(f"{ch}→+1(word-initial or >VCC,schwa survives)")
            continue

        # Ohala fires → Pandey
        apply_pandey = (ru['type'] == 'consonant' and (ru['matra'] is None or lha))
        if apply_pandey:
            c1, c2 = u['base'], ru['base']
            if (c1, c2) not in _VALID_ONSETS:
                syllables += 1; lv = i
                decisions.append(
                    f"{ch}→+1(Ohala fires,Pandey blocks: {c1}+{c2} invalid onset)")
                continue

        decisions.append(f"{ch}→0(Ohala fires,schwa deleted)")

    return units, decisions, max(1, syllables)


def hindi_line_steps(text: str) -> int:
    """Count syllables in a Hindi line, printing per-word steps."""
    total = 0
    words = text.strip().split()
    for word in words:
        clean = re.sub(r'[^\u0900-\u097F]', '', word)
        if not clean:
            continue
        units, decisions, count = hindi_word_steps(clean)
        print(f"      '{word}'")
        print(f"        units    : {[u['base'] for u in units]}")
        print(f"        decisions: {decisions}")
        print(f"        syllables: {count}")
        total += count
    return total


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

W = 68

def print_header(title, subtitle):
    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"  {subtitle}")
    print(f"{'═'*W}")


def run_english(sentences: List[str]):
    print_header(
        "ENGLISH PIPELINE",
        "text → espeak-ng → IPA → collapse diphthongs → count nuclei"
    )
    for i, text in enumerate(sentences, 1):
        print(f"\n  [{i:02d}] '{text}'")
        print(f"  {'─'*64}")
        ipa_raw, ipa_coll, nuclei, count = english_steps(text)
        print(f"      STEP 1  raw IPA      : {ipa_raw}")
        found = [d for d in _DIPHTHONGS_EN if d in ipa_raw]
        if found:
            print(f"      STEP 2  diphthongs   : {found} → each collapsed to one token")
            print(f"              collapsed IPA: {ipa_coll}")
        else:
            print(f"      STEP 2  diphthongs   : none found")
        print(f"      STEP 3  nuclei       : {nuclei}")
        print(f"      RESULT  syllables    : {count}")


def run_hindi(sentences: List[str]):
    print_header(
        "HINDI PIPELINE",
        "Devanagari → parse units → Ohala (1983) + Pandey (1990) → count nuclei"
    )
    for i, (deva, label) in enumerate(sentences, 1):
        print(f"\n  [{i:02d}] '{label}'")
        print(f"  {'─'*64}")
        total = hindi_line_steps(deva)
        print(f"      RESULT  syllables    : {total}")


# ══════════════════════════════════════════════════════════════════════════════
# SENTENCES
# 20 each — diverse: single words, short phrases, full lyric lines
# Covering: standard words, contractions, truncated forms (EN)
#           halant conjuncts, anusvara, chandrabindu, mixed (HI)
# ══════════════════════════════════════════════════════════════════════════════

ENGLISH_SENTENCES = [
    "I love you",
    "Never gonna give you up",
    "Lose yourself in the music",
    "Hold me close and never let go",
    "Dancing in the moonlight",
    "Every breath you take",
    "Runnin' through the fire",
    "Lovin' every minute of it",
    "Nothin' else matters tonight",
    "Fire and desire",
    "She's a prayer warrior",
    "Ooh baby don't you cry",
    "Cause every time we touch I feel the rush",
    "Ain't no sunshine when she's gone",
    "We found love in a hopeless place",
    "Higher and higher baby",
    "I will always love you",
    "Baby one more time",
    "Rolling in the deep",
    "Somewhere only we know",
]

HINDI_SENTENCES = [
    ("\u0926\u093F\u0932", "दिल"),
    ("\u092A\u094D\u092F\u093E\u0930", "प्यार"),
    ("\u091C\u093C\u093F\u0902\u0926\u0917\u0940", "ज़िंदगी"),
    ("\u0939\u092E\u0947\u0936\u093E", "हमेशा"),
    ("\u0924\u0941\u091D\u0938\u0947 \u092E\u0941\u091D\u0947 \u092A\u094D\u092F\u093E\u0930 \u0939\u0948",
     "तुझसे मुझे प्यार है"),
    ("\u0924\u0941\u092E \u0939\u0940 \u0939\u094B", "तुम ही हो"),
    ("\u0926\u093F\u0932 \u092E\u0947\u0902 \u0924\u0947\u0930\u0947", "दिल में तेरे"),
    ("\u0939\u092E\u0947\u0936\u093E \u0926\u093F\u0932 \u092E\u0947\u0902 \u0930\u0939\u0924\u0947 \u0939\u094B",
     "हमेशा दिल में रहते हो"),
    ("\u091C\u093C\u093F\u0902\u0926\u0917\u0940 \u0917\u0941\u0932\u091C\u093C\u093E\u0930 \u0939\u0948",
     "ज़िंदगी गुलज़ार है"),
    ("\u091A\u093E\u0901\u0926 \u0915\u0940 \u0930\u094B\u0936\u0928\u0940 \u092E\u0947\u0902 \u0928\u093E\u091A\u0942\u0902",
     "चाँद की रोशनी में नाचूं"),
    ("\u0915\u092D\u0940 \u0915\u092D\u0940 \u092E\u0947\u0930\u0947 \u0926\u093F\u0932 \u092E\u0947\u0902",
     "कभी कभी मेरे दिल में"),
    ("\u092C\u094B\u0932 \u0928\u093E \u0939\u0932\u094D\u0915\u0947 \u0939\u0932\u094D\u0915\u0947",
     "बोल ना हल्के हल्के"),
    ("\u092E\u0947\u0930\u0947 \u0938\u092A\u0928\u094B\u0902 \u0915\u0940 \u0930\u093E\u0928\u0940",
     "मेरे सपनों की रानी"),
    ("\u092A\u0939\u0932\u093E \u0928\u0936\u093E \u092A\u0939\u0932\u093E \u0916\u0941\u092E\u093E\u0930",
     "पहला नशा पहला खुमार"),
    ("\u0924\u0947\u0930\u0947 \u092C\u093F\u0928\u093E \u091C\u093C\u093F\u0902\u0926\u0917\u0940 \u0938\u0947",
     "तेरे बिना ज़िंदगी से"),
    ("\u0938\u0941\u0928 \u0930\u0939\u093E \u0939\u0948 \u0928\u093E \u0924\u0942",
     "सुन रहा है ना तू"),
    ("\u0924\u0947\u0930\u0947 \u0928\u0948\u0928\u093E", "तेरे नैना"),
    ("\u0930\u094B\u0936\u0928\u0940", "रोशनी"),
    ("\u0938\u0902\u0917\u0940\u0924", "संगीत"),
    ("\u092E\u0947\u0930\u093E \u0926\u093F\u0932 \u092D\u0940 \u0915\u093F\u0924\u0928\u093E \u092A\u093E\u0917\u0932 \u0939\u0948",
     "मेरा दिल भी कितना पागल है"),
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    run_english(ENGLISH_SENTENCES)
    run_hindi(HINDI_SENTENCES)
    print()