import os
import random
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
import pandas as pd

from semantic_translator import get_unique_hindi_lines, get_or_create_embeddings
from synonym_swapper import SynonymSwapper


# ==============================
# PATH TO ENGLISH SONG DATASET
# ==============================
script_dir = os.path.dirname(os.path.abspath(__file__))
ENGLISH_DATASET_PATH = os.path.join(script_dir, "..", "English_Lyrics_Database", "csv")


def clean_lyrics_text(text):
    """
    Converts raw lyric blob into clean list of lines
    """
    if pd.isna(text):
        return []

    text = str(text)

    # normalize weird formatting
    text = text.replace("\r", "\n")
    text = text.replace(" / ", "\n")
    text = text.replace(" . ", ".\n")

    # better splitting
    raw_lines = re.split(r"[.,!?]\s+|\n", text)

    lines = []
    for line in raw_lines:
        line = line.strip().lower()

        if not line:
            continue
        if len(line) < 2:
            continue
        if line.isdigit():
            continue
        if "[" in line and "]" in line:
            continue

        words = line.split()

        # 🔥 break long lines into smaller chunks
        if len(words) > 12:
            for i in range(0, len(words), 8):
                chunk = " ".join(words[i:i+8])
                if len(chunk.split()) > 1:
                    lines.append(chunk)
        else:
            lines.append(line)

    return lines


def pick_random_song():
    """
    Picks ONE random song from ONE random artist CSV
    Returns:
        song_lines (list[str]), song_name (str)
    """

    csv_files = [
        f for f in os.listdir(ENGLISH_DATASET_PATH)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise ValueError("❌ No CSV files found")

    chosen_csv = random.choice(csv_files)
    path = os.path.join(ENGLISH_DATASET_PATH, chosen_csv)

    print(f"\n📂 Loading artist file: {chosen_csv}")

    df = pd.read_csv(path)

    # detect lyrics column
    lyrics_col = None
    for col in df.columns:
        if "lyric" in col.lower():
            lyrics_col = col
            break

    if lyrics_col is None:
        raise ValueError(f"❌ No lyrics column in {chosen_csv}")

    # remove empty lyrics rows
    df = df.dropna(subset=[lyrics_col])
    df = df[df[lyrics_col].astype(str).str.len() > 20]

    # pick ONE random song row
    row = df.sample(1).iloc[0]

    raw_lyrics = row[lyrics_col]

    # try to extract song name safely
    song_name = None
    for key in ["song", "title", "name"]:
        if key in df.columns:
            song_name = row.get(key)
            break

    if song_name is None or pd.isna(song_name):
        song_name = chosen_csv.replace(".csv", "")

    print(f"🎧 Selected Song: {song_name}")

    # convert lyrics → clean lines
    song_lines = clean_lyrics_text(raw_lyrics)

    # final safety filter
    song_lines = [
    l for l in song_lines
    if len(l.split()) >= 1
    ]

    if len(song_lines) == 0:
        print("⚠️ Empty lyrics after cleaning — retrying song")
        return pick_random_song()

    print("DEBUG: number of lines =", len(song_lines))
    print(song_lines[:5])

    return song_lines, song_name

# ==============================
# BUILD CHUNK DICTIONARY
# ==============================
def build_chunk_dictionary(lines, max_n=4, min_freq=5):
    counter = Counter()

    for line in lines:
        if isinstance(line, list):
            continue  # safety guard

        words = str(line).lower().split()

        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                chunk = " ".join(words[i:i+n])
                counter[chunk] += 1

    chunk_dict = {c for c, f in counter.items() if f >= min_freq}

    print(f"✅ Chunk dictionary size: {len(chunk_dict)}")
    return chunk_dict


# ==============================
# SPLIT INTO CHUNKS (LONGEST MATCH)
# ==============================
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


# ==============================
# INITIALIZE MODEL + DATABASE
# ==============================
print("Loading model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

print("Loading Hindi lines...")
lines = get_unique_hindi_lines()

print("Loading embeddings...")
db_lines, db_embeddings = get_or_create_embeddings(model, lines)

print("Initializing swapper...")
swapper = SynonymSwapper(model, db_lines, db_embeddings)

print("Building chunk dictionary...")
chunk_dict = build_chunk_dictionary(lines, min_freq=2)
print("Sample chunks:", list(chunk_dict)[:10])


# ==============================
# TRANSLATE ONE LINE (FIXED)
# ==============================
def translate_line(english_line):
    # 🔥 use TOP-K instead of single best
    candidates = swapper.top_k(english_line, k=5)

    best_line = None
    best_score = -1
    best_syllables = 0

    for c in candidates:
        hindi_line = c['line']

        # split into chunks
        chunks = split_into_chunks(hindi_line, chunk_dict)

        # keep short (song-like)
        MAX_CHUNKS = 3
        chunks = chunks[:MAX_CHUNKS]

        final_line = " ".join(chunks)

        # scoring: semantic + shorter is better
        score = c['semantic_score'] - 0.05 * len(chunks)

        if score > best_score:
            best_score = score
            best_line = final_line
            best_syllables = c['hi_syllables']

    if not best_line:
        return {
            "english": english_line,
            "hindi": "[NO MATCH]",
            "syllables": 0,
            "score": 0
        }

    return {
        "english": english_line,
        "hindi": best_line,
        "syllables": best_syllables,
        "score": best_score
    }


# ==============================
# TRANSLATE FULL SONG
# ==============================
def translate_song(song_lines, name):
    print(f"\n🎧 Selected Song: {name}")
    print("\n🎵 TRANSLATED SONG:\n")

    if not song_lines:
        print("❌ No valid lyric lines to translate")
        return

    for line in song_lines:
        line = line.strip()

        # skip super long garbage lines
        #if len(line.split()) > 20:
        #    continue

        result = translate_line(line)

        print("EN :", result["english"])
        print("HI :", result["hindi"])
        print(f"SYL: {result['syllables']} | SCORE: {result['score']:.4f}")
        print("-" * 50)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    song_lines, name = pick_random_song()
    translate_song(song_lines, name)