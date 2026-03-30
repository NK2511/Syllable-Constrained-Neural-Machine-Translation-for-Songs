
import os
import glob
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Step 1: Model Setup
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
CACHE_FILE = "hindi_lyrics_embeddings.pt"
DB_PATH = r"c:\Desktop\Python\Amrita_Vishwa_Vidyapeetam\Sem_6\Syllable-Constrained-Neural-Machine-Translation-for-Songs\Hindi_Lyrics_Database\scraped_lyrics_cleaned"

def get_unique_hindi_lines():
    """Load all unique lines from the lyrics database."""
    print("Reading Hindi database...")
    txt_files = glob.glob(os.path.join(DB_PATH, "*.txt"))
    unique_lines = set()
    for file in txt_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    clean_line = line.strip()
                    if clean_line and len(clean_line) > 5: # Filter noise
                        unique_lines.add(clean_line)
        except:
            continue
    print(f"Loaded {len(unique_lines)} unique Hindi lines.")
    return sorted(list(unique_lines))

def get_or_create_embeddings(model, lines):
    """Calculate or load pre-built embeddings for the database."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached embeddings from {CACHE_FILE}...")
        data = torch.load(CACHE_FILE)
        
        # Auto-detect changes: Compare cached lines with current DB lines
        if data.get('lines') == lines:
            print("Cache is up to date.")
            return data['lines'], data['embeddings']
        else:
            print("Change detected in Lyrics Database! Re-calculating embeddings...")
    
    print("Calculating embeddings (this may take a few minutes on the first run)...")
    embeddings = model.encode(lines, convert_to_tensor=True, show_progress_bar=True)
    
    # Cache for future use
    torch.save({'lines': lines, 'embeddings': embeddings}, CACHE_FILE)
    print(f"Successfully cached embeddings to {CACHE_FILE}.")
    return lines, embeddings

def run_translator():
    model = SentenceTransformer(model_name)
    lines = get_unique_hindi_lines()
    all_lines, db_embeddings = get_or_create_embeddings(model, lines)

    print("\n=== Semantic Bollywood Translator ===")
    print("Type an English line, and I will find the closest Hindi 'Purport' from our database.")

    while True:
        eng_line = input("\nEnter English line: ")
        if not eng_line: break

        # 1. Encode target English line
        target_embedding = model.encode(eng_line, convert_to_tensor=True)

        # 2. Fast similarity search (Matrix multiplication)
        similarities = util.pytorch_cos_sim(target_embedding, db_embeddings)[0]
        
        # 3. Get top 5 matches
        top_results = torch.topk(similarities, k=5)

        print("\n" + "="*60)
        print(f"TOP HINDI 'PURPORTS' FOR: '{eng_line}'")
        print("="*60)
        for score, idx in zip(top_results[0], top_results[1]):
            print(f"[{score:.4f}] {all_lines[idx.item()]}")
        print("="*60)

        cont = input("\nTry another? (y/n): ").lower()
        if cont != 'y':
            break

if __name__ == "__main__":
    run_translator()
