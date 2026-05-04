import re
from collections import Counter

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_line(text):
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    return text.strip()


# -----------------------------
# BUILD CHUNK DICTIONARY
# -----------------------------
def build_chunk_dictionary(lines, max_n=4, min_freq=5):
    counter = Counter()

    for line in lines:
        words = clean_line(line).split()

        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                chunk = " ".join(words[i:i+n])
                counter[chunk] += 1

    # Keep only frequent chunks
    chunk_dict = {c for c, f in counter.items() if f >= min_freq}

    print(f"✅ Built {len(chunk_dict)} chunks")
    return chunk_dict


# -----------------------------
# LONGEST MATCH CHUNK SPLIT
# -----------------------------
def split_into_chunks(line, chunk_dict, max_n=4):
    words = clean_line(line).split()
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