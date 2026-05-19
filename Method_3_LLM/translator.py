import os
import sys
import re
import io
import contextlib
import random
import pickle
import pathlib
import google.generativeai as genai
import time
import google.api_core.exceptions as google_exceptions


# ==========================================
# Step 1: Import SyllableCounter_final.py
# ==========================================
sys.path.insert(0, ".")
try:
    from SyllableCounter_final import english_steps, hindi_line_steps
except ImportError:
    print("ERROR: SyllableCounter_final.py not found in the current directory.")
    print("Please ensure it is copied into the LLM_Trials folder.")
    sys.exit(1)

# ==========================================
# Step 2: Load Hindi Lyrics Database
# ==========================================
def load_hindi_db(folder_path):
    lines = []
    if not os.path.exists(folder_path):
        print(f"ERROR: Database folder '{folder_path}' not found in LLM_Trials.")
        sys.exit(1)
        
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    # Keep only lines with Devanagari characters
                    if re.search(r"[\u0900-\u097F]", line) and len(line) > 3:
                        lines.append(line)
    return lines

# ==========================================
# Step 3: Count Syllables Silently
# ==========================================
def count_hindi_silent(text):
    """Run hindi_line_steps() but suppress its print output."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        count = hindi_line_steps(text)
    return count

# ==========================================
# Step 4: Initialise Gemini Client
# ==========================================
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set it in your terminal before running this script.")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-flash-latest")

# ==========================================
# Step 5: Count English Syllables
# ==========================================
def get_english_count(line):
    _, _, _, count = english_steps(line)
    return count

# ==========================================
# Step 6: Sample Few-Shot Examples
# ==========================================
def sample_few_shot(db_with_counts, target_n, k=5, window=2):
    """Return k Hindi lines with syllable count in [target_n-window, target_n+window]."""
    pool = [(line, cnt) for line, cnt in db_with_counts
            if abs(cnt - target_n) <= window]
    if len(pool) < k:
        pool = sorted(db_with_counts, key=lambda x: abs(x[1] - target_n))[:k*3]
    return random.sample(pool, min(k, len(pool)))

# ==========================================
# Step 7: Build LLM Prompt
# ==========================================
def build_prompt(english_line, target_n, few_shot_examples):
    system = (
        "You are a Bollywood lyricist in the style of Gulzar and Javed Akhtar. "
        "You write poetic Hindi that is singable, emotional, and uses "
        "literary vocabulary. Never use casual spoken Hindi."
    )
    
    examples_text = "\n".join(
        f"  - {line}  ({cnt} syllables)"
        for line, cnt in few_shot_examples
    )
    
    task = (
        f"Translate this English lyric line into Hindi with EXACTLY {target_n} syllables.\n"
        f"English: {english_line}\n\n"
        f"Here are real Bollywood lyric lines with a similar syllable count for style reference:\n"
        f"{examples_text}\n\n"
        f"Rules:\n"
        f"  1. Output ONLY the Hindi line in Devanagari script. No transliteration.\n"
        f"  2. The translation must have EXACTLY {target_n} syllables.\n"
        f"  3. Preserve the emotional meaning of the English line.\n"
        f"  4. Sound poetic and singable — like a Bollywood song lyric.\n"
        f"  5. Do not include the syllable count or any other text in your output."
    )
    return system + "\n\n" + task

# ==========================================
# Step 8: Verify Syllable Count
# ==========================================
def get_hindi_syllable_count(hindi_text):
    return count_hindi_silent(hindi_text)

def call_llm(prompt):
    delay = 6
    for attempt in range(6):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except google_exceptions.ResourceExhausted as e:
            print(f"\n  [Rate Limit] Exceeded quota. Retrying in {delay}s... (Error: {e.message})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        except Exception as e:
            if "ResourceExhausted" in str(e) or "429" in str(e):
                print(f"\n  [Rate Limit] Exceeded quota. Retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise e
    raise Exception("Exhausted all retries due to Gemini Rate Limits (429).")

# ==========================================
# Step 9: Retry Loop with Feedback
# ==========================================
MAX_RETRIES = 4

def translate(english_line, db_with_counts):
    target_n = get_english_count(english_line)
    few_shot = sample_few_shot(db_with_counts, target_n)
    prompt = build_prompt(english_line, target_n, few_shot)
    
    best = None
    best_diff = float("inf")
    
    for attempt in range(1, MAX_RETRIES + 1):
        hindi = call_llm(prompt)
        
        # Devanagari validation check
        if not re.search(r"[\u0900-\u097F]", hindi):
            print(f"  Attempt {attempt}: [Romanised/Invalid] {hindi}")
            prompt += "\nIMPORTANT: Respond ONLY in Devanagari script."
            continue
            
        actual = get_hindi_syllable_count(hindi)
        diff = abs(actual - target_n)
        
        print(f"  Attempt {attempt}: [{actual} syllables] {hindi}")
        
        if diff < best_diff:
            best_diff = diff
            best = hindi
            
        if actual == target_n:
            return hindi, True   # success
            
        direction = "shorter" if actual > target_n else "longer"
        prompt += (
            f"\n\nYour last output was: {hindi}\n"
            f"That had {actual} syllables. I need exactly {target_n}.\n"
            f"Please rewrite it to be {direction} by {diff} syllable(s)."
        )
        
    print(f"  WARNING: Could not match {target_n} syllables. Best was {best_diff} off.")
    return best, False

# ==========================================
# Step 10: Main Loop
# ==========================================
if __name__ == "__main__":
    print("========================================")
    print(" Hindi Lyric Translator — LLM Pipeline")
    print("========================================")
    print("Loading Hindi database... ", end="", flush=True)
    
    # Implementing the cache optimization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CACHE = pathlib.Path(os.path.join(script_dir, "db_cache.pkl"))
    
    cached_loaded = False
    if CACHE.exists():
        try:
            DB_WITH_COUNTS = pickle.loads(CACHE.read_bytes())
            if len(DB_WITH_COUNTS) > 0:
                print(f"loaded from cache! ({len(DB_WITH_COUNTS)} lines)")
                cached_loaded = True
        except Exception:
            pass

    if not cached_loaded:
        HINDI_DB = load_hindi_db(os.path.join(script_dir, "Hindi_Lyrics_Database", "scraped_lyrics_cleaned"))
        print(f"parsing {len(HINDI_DB)} lines (this may take a minute)... ", end="", flush=True)
        DB_WITH_COUNTS = [(l, count_hindi_silent(l)) for l in HINDI_DB]
        CACHE.write_bytes(pickle.dumps(DB_WITH_COUNTS))
        print("done and cached!")

    while True:
        eng = input("\nEnglish line (or 'q' to quit): ").strip()
        if eng.lower() == 'q':
            break
        if not eng:
            continue
            
        try:
            result, matched = translate(eng, DB_WITH_COUNTS)
            if result:
                status = "MATCHED" if matched else "CLOSEST (flagged)"
                print(f"\n  [{status}] {result}")
        except Exception as e:
            print(f"  [ERROR] Translation failed: {e}")
