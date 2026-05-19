import os
import sys
import csv
import torch
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import io
import contextlib

# Add current and parent folder to path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, ".."))

try:
    from SyllableCounter_final import english_steps, hindi_line_steps
    import semantic_translator
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def get_english_count(line):
    _, _, _, count = english_steps(line)
    return count

def count_hindi_silent(text):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        count = hindi_line_steps(text)
    return count

# =====================================================================
# SYLLABLE-CONSTRAINED SEMANTIC RETRIEVAL ENGINE
# =====================================================================
def get_retrieval_options(english_line, model, db_lines, db_embeddings, target_n, num_options=3):
    """
    Finds top matches in the Hindi database by balancing:
    1. Semantic similarity (cosine similarity from sentence-transformers)
    2. Syllable count alignment (penalizing deviations from target_n)
    """
    # 1. Encode English line
    target_embedding = model.encode(english_line, convert_to_tensor=True)
    
    # 2. Calculate cosine similarity against all database entries
    similarities = util.pytorch_cos_sim(target_embedding, db_embeddings)[0]
    
    # Get top 200 semantic matches
    top_k_semantic = min(200, len(db_lines))
    scores, indices = torch.topk(similarities, k=top_k_semantic)
    
    candidates = []
    for score, idx in zip(scores, indices):
        h_line = db_lines[idx.item()]
        h_syl = count_hindi_silent(h_line)
        diff = abs(h_syl - target_n)
        
        # Combined Score = Semantic similarity - (syllable difference penalty)
        # We use a penalty weight of 0.035 per syllable difference
        combined_score = score.item() - 0.035 * diff
        
        candidates.append({
            "hindi": h_line,
            "syllables": h_syl,
            "diff": diff,
            "semantic_score": score.item(),
            "combined_score": combined_score
        })
        
    # Sort candidates by combined score in descending order
    candidates = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)
    
    # Filter out near-duplicates to ensure option variety
    unique_options = []
    seen = set()
    for cand in candidates:
        normalized = re.sub(r'\s+', '', cand['hindi'])
        if normalized not in seen:
            seen.add(normalized)
            unique_options.append(cand)
            if len(unique_options) >= num_options:
                break
                
    return unique_options

def process_text_file(input_txt, csv_output):
    print(f"\nReading {input_txt}...")
    with open(input_txt, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and len(line.strip()) > 2]
        
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Loading database lines...")
    db_lines = semantic_translator.get_unique_hindi_lines()
    
    print("Loading pre-calculated embeddings...")
    # Point to the embeddings cache in Method_1_Retrieval
    cache_path = os.path.join(script_dir, "hindi_lyrics_embeddings.pt")
    if not os.path.exists(cache_path):
        print(f"Embeddings cache not found at {cache_path}. Creating a new one...")
    db_lines, db_embeddings = semantic_translator.get_or_create_embeddings(model, db_lines)
    
    results = []
    print("\nRetrieving syllable-constrained translation purports (offline)...")
    for i, eng_line in enumerate(lines):
        target_syl = get_english_count(eng_line)
        print(f"  Line {i+1}/{len(lines)}: {eng_line} (Target: {target_syl} syl)")
        
        options = get_retrieval_options(eng_line, model, db_lines, db_embeddings, target_syl)
        results.append({
            "english": eng_line,
            "target_syllables": target_syl,
            "options": options
        })
        
    # Save options to CSV
    with open(csv_output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = ["English_Line", "Target_Syllables"]
        for i in range(3):
            header.extend([f"Option_{i+1}_Hindi", f"Option_{i+1}_Syllables", f"Option_{i+1}_SemanticScore"])
        writer.writerow(header)
        
        for r in results:
            row = [r['english'], r['target_syllables']]
            for opt in r['options']:
                row.extend([opt['hindi'], opt['syllables'], f"{opt['semantic_score']:.4f}"])
            writer.writerow(row)
            
    print(f"\nSaved translation database to {csv_output}")
    return results

def interactive_selection(results, final_txt_output):
    print("\n" + "="*50)
    print(" 🎯 INTERACTIVE SEMANTIC RETRIEVAL SELECTION 🎯")
    print("="*50)
    
    selected_lines = []
    for i, r in enumerate(results):
        print(f"\nLine {i+1}: {r['english']} (Target: {r['target_syllables']} syllables)")
        
        valid_choices = []
        for j, opt in enumerate(r['options']):
            diff = opt['syllables'] - r['target_syllables']
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            match_str = "✅ MATCH" if diff == 0 else f"❌ {diff_str}"
            
            # Show options with semantic score and syllable match status
            print(f"  [{j+1}] {opt['hindi']} ({opt['syllables']} syl | Sim: {opt['semantic_score']:.3f} | {match_str})")
            valid_choices.append(str(j+1))
            
        print(f"  [4] Type your own custom translation/purport")
        
        while True:
            choice = input(f"Choose option (1-3) or 4 to type your own: ").strip()
            if choice in valid_choices:
                selected = r['options'][int(choice)-1]['hindi']
                selected_lines.append(selected)
                break
            elif choice == '4':
                custom = input("Enter your custom Hindi line in Devanagari: ").strip()
                selected_lines.append(custom)
                break
            else:
                print("Invalid choice. Try again.")
                
    with open(final_txt_output, 'w', encoding='utf-8') as f:
        for line in selected_lines:
            f.write(line + "\n")
            
    print(f"\nFinal selections saved to {final_txt_output}")
    return selected_lines

def generate_tts_audio(hindi_lines, output_wav):
    print("\nGenerating spoken TTS audio for the selected lines...")
    try:
        from pydub import AudioSegment
    except ImportError:
        print("Please install pydub: pip install pydub")
        return
        
    combined_audio = AudioSegment.silent(duration=500)
    
    for i, line in enumerate(hindi_lines):
        print(f"  Synthesizing line {i+1}...")
        temp_mp3 = f"temp_{i}.mp3"
        tts = gTTS(text=line, lang='hi', slow=False)
        tts.save(temp_mp3)
        
        audio = AudioSegment.from_mp3(temp_mp3)
        combined_audio += audio + AudioSegment.silent(duration=800)
        os.remove(temp_mp3)
        
    combined_audio.export(output_wav, format="wav")
    print(f"✅ Full spoken audio saved to {output_wav}")

if __name__ == "__main__":
    # Check for shape_of_you_lyrics.txt in different directories
    possible_lyrics_paths = [
        os.path.join(script_dir, "..", "Method_3_LLM", "shape_of_you_lyrics.txt"),
        os.path.join(script_dir, "shape_of_you_lyrics.txt"),
        os.path.join(script_dir, "..", "shape_of_you_lyrics.txt")
    ]
    
    input_txt = None
    for p in possible_lyrics_paths:
        if os.path.exists(p):
            input_txt = p
            break
            
    if not input_txt:
        print("Error: Could not find shape_of_you_lyrics.txt.")
        print(f"Please place it at: {possible_lyrics_paths[0]}")
        sys.exit(1)
        
    csv_output = os.path.join(script_dir, "translation_options.csv")
    final_txt_output = os.path.join(script_dir, "final_hindi_lyrics.txt")
    audio_output = os.path.join(script_dir, "..", "Synthesize", "final_spoken_lyrics.wav")
    
    results = process_text_file(input_txt, csv_output)
    print("\n✅ Batch translation search complete!")
    print(f"Please inspect the generated options in: {csv_output}")
    print("You can now run the synthesis script to choose options and match them to the song's timing.")
