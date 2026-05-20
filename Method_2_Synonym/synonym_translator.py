import os
import random
import re
from collections import Counter
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from semantic_translator import (
    get_unique_hindi_lines,
    get_or_create_embeddings
)

from Method_2_Synonym.synonym_swapper import SynonymSwapper


# =========================================================
# PATHS
# =========================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

ENGLISH_DATASET_PATH = os.path.join(
    script_dir,
    "..",
    "English_Lyrics_Database",
    "csv"
)


# =========================================================
# GLOBAL MEMORY
# =========================================================
used_lines = set()


# =========================================================
# BAD / NOISY WORDS
# =========================================================
BAD_WORDS = {
    "yeah", "uh", "ooo", "oooh", "oh",
    "ayy", "huh", "la", "na", "baby",
    "shawty", "yo", "yah", "woo",
    "nicki", "minaj", "jay", "sean",
    "feat", "ft", "dj", "mc"
}


# =========================================================
# CLEAN ENGLISH LINE
# =========================================================
def clean_english_line(line):

    line = line.lower()

    # remove punctuation
    line = re.sub(r"[^a-zA-Z0-9\s]", " ", line)

    words = []

    for w in line.split():

        if w in BAD_WORDS:
            continue

        if len(w) <= 1:
            continue

        words.append(w)

    return " ".join(words)


# =========================================================
# CLEAN LYRICS TEXT
# =========================================================
def clean_lyrics_text(text):

    if pd.isna(text):
        return []

    text = str(text)

    text = text.replace("\r", "\n")
    text = text.replace(" / ", "\n")
    text = text.replace(" . ", ".\n")

    raw_lines = re.split(r"[.,!?]\s+|\n", text)

    lines = []

    for line in raw_lines:

        line = clean_english_line(line)

        if not line:
            continue

        if len(line) < 2:
            continue

        if line.isdigit():
            continue

        words = line.split()

        # split long lines
        if len(words) > 12:

            for i in range(0, len(words), 8):

                chunk = " ".join(words[i:i+8])

                if len(chunk.split()) > 1:
                    lines.append(chunk)

        else:
            lines.append(line)

    return lines


# =========================================================
# RANDOM SONG PICKER
# =========================================================
def pick_random_song():

    csv_files = [
        f for f in os.listdir(ENGLISH_DATASET_PATH)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise ValueError("❌ No CSV files found")

    chosen_csv = random.choice(csv_files)

    path = os.path.join(
        ENGLISH_DATASET_PATH,
        chosen_csv
    )

    print(f"\n📂 Loading artist file: {chosen_csv}")

    df = pd.read_csv(path)

    # detect lyrics column
    lyrics_col = None

    for col in df.columns:

        if "lyric" in col.lower():
            lyrics_col = col
            break

    if lyrics_col is None:
        raise ValueError("❌ No lyrics column found")

    df = df.dropna(subset=[lyrics_col])

    df = df[
        df[lyrics_col].astype(str).str.len() > 20
    ]

    row = df.sample(1).iloc[0]

    raw_lyrics = row[lyrics_col]

    # song name
    song_name = None

    for key in ["song", "title", "name"]:

        if key in df.columns:
            song_name = row.get(key)
            break

    if song_name is None or pd.isna(song_name):
        song_name = chosen_csv.replace(".csv", "")

    print(f"🎧 Selected Song: {song_name}")

    song_lines = clean_lyrics_text(raw_lyrics)

    if len(song_lines) == 0:

        print("⚠️ Empty lyrics after cleaning")
        return pick_random_song()

    print("DEBUG lines:", len(song_lines))
    print(song_lines[:5])

    return song_lines, song_name


# =========================================================
# BUILD CHUNK DICTIONARY
# =========================================================
def build_chunk_dictionary(lines, max_n=4, min_freq=2):

    counter = Counter()

    for line in lines:

        words = str(line).lower().split()

        for n in range(1, max_n + 1):

            for i in range(len(words) - n + 1):

                chunk = " ".join(words[i:i+n])

                counter[chunk] += 1

    chunk_dict = {
        c for c, f in counter.items()
        if f >= min_freq
    }

    print(f"✅ Chunk dictionary size: {len(chunk_dict)}")

    return chunk_dict


# =========================================================
# SPLIT INTO CHUNKS
# =========================================================
def split_into_chunks(line, chunk_dict, max_n=4):

    words = str(line).lower().split()

    i = 0

    chunks = []

    while i < len(words):

        found = None

        for n in range(max_n, 0, -1):

            if i + n <= len(words):

                candidate = " ".join(words[i:i+n])

                if candidate in chunk_dict:

                    found = candidate
                    break

        if found:

            chunks.append(found)

            i += len(found.split())

        else:

            chunks.append(words[i])

            i += 1

    return chunks


# =========================================================
# EMOTION DETECTOR
# =========================================================
def detect_emotion(line):

    line = line.lower()

    love_words = {
        "love", "baby", "kiss",
        "heart", "romance", "girl"
    }

    sad_words = {
        "cry", "alone", "pain",
        "hurt", "sad", "broken"
    }

    party_words = {
        "dance", "club", "party",
        "night", "drink"
    }

    if any(w in line for w in love_words):
        return "romantic"

    if any(w in line for w in sad_words):
        return "sad"

    if any(w in line for w in party_words):
        return "party"

    return "neutral"


# =========================================================
# EMOTION BONUS
# =========================================================
def emotion_bonus(eng_emotion, hindi_line):

    hindi_line = hindi_line.lower()

    romantic = [
        "प्यार", "इश्क", "मोहब्बत",
        "दिल", "सनम"
    ]

    sad = [
        "दर्द", "आँसू",
        "तन्हा", "जुदाई"
    ]

    party = [
        "नाच", "दारू",
        "रात", "झूम"
    ]

    if eng_emotion == "romantic":

        if any(w in hindi_line for w in romantic):
            return 0.08

    if eng_emotion == "sad":

        if any(w in hindi_line for w in sad):
            return 0.08

    if eng_emotion == "party":

        if any(w in hindi_line for w in party):
            return 0.08

    return 0


# =========================================================
# INITIALIZE MODEL
# =========================================================
print("Loading model...")

model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

print("Loading Hindi lines...")

lines = get_unique_hindi_lines()

print("Loading embeddings...")

db_lines, db_embeddings = get_or_create_embeddings(
    model,
    lines
)

print("Initializing synonym swapper...")

swapper = SynonymSwapper(
    model,
    db_lines,
    db_embeddings
)

print("Building chunk dictionary...")

chunk_dict = build_chunk_dictionary(
    lines,
    min_freq=2
)

print("Sample chunks:")
print(list(chunk_dict)[:10])


# =========================================================
# TRANSLATE ONE LINE
# =========================================================
def translate_line(english_line):

    print(f"\n🔍 Processing: {english_line}")

    cleaned_line = clean_english_line(
        english_line
    )

    if not cleaned_line:

        return {
            "english": english_line,
            "hindi": "[SKIPPED]",
            "score": 0,
            "syllables": 0
        }

    eng_emotion = detect_emotion(
        cleaned_line
    )

    # synonym-expanded retrieval
    candidates = swapper.top_k(
        cleaned_line,
        k=10
    )

    best_line = None
    best_score = -999
    best_syllables = 0

    for c in candidates:

        hindi_line = c["line"]

        semantic_score = c["semantic_score"]

        chunks = split_into_chunks(
            hindi_line,
            chunk_dict
        )

        # dynamic chunk limit
        word_count = len(cleaned_line.split())

        if word_count <= 4:
            MAX_CHUNKS = 2

        elif word_count <= 8:
            MAX_CHUNKS = 4

        else:
            MAX_CHUNKS = 6

        chunks = chunks[:MAX_CHUNKS]

        final_line = " ".join(chunks)

        score = semantic_score

        # shorter is better
        score -= 0.03 * len(chunks)

        # emotion matching
        score += emotion_bonus(
            eng_emotion,
            final_line
        )

        # diversity penalty
        if final_line in used_lines:
            score -= 0.15

        # exact keyword overlap
        eng_words = set(cleaned_line.split())

        overlap = 0

        for w in eng_words:

            if w in hindi_line.lower():
                overlap += 1

        score += overlap * 0.02

        if score > best_score:

            best_score = score
            best_line = final_line
            best_syllables = c["hi_syllables"]

    if not best_line:

        return {
            "english": english_line,
            "hindi": "[NO MATCH]",
            "score": 0,
            "syllables": 0
        }

    used_lines.add(best_line)

    return {
        "english": english_line,
        "hindi": best_line,
        "score": best_score,
        "syllables": best_syllables
    }


# =========================================================
# TRANSLATE FULL SONG
# =========================================================
def translate_song(song_lines, name):

    print(f"\n🎧 Selected Song: {name}")

    print("\n🎵 TRANSLATED SONG:\n")

    if not song_lines:

        print("❌ No valid lyric lines")
        return

    for line in song_lines:

        line = line.strip()

        if not line:
            continue

        result = translate_line(line)

        print("EN :", result["english"])

        print("HI :", result["hindi"])

        print(
            f"SYL: {result['syllables']} | "
            f"SCORE: {result['score']:.4f}"
        )

        print("-" * 50)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    song_lines, name = pick_random_song()

    translate_song(song_lines, name)