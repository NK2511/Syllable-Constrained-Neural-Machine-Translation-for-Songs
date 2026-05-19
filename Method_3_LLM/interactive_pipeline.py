import os
import sys
import csv
import pickle
import pathlib
import google.generativeai as genai
from gtts import gTTS
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from SyllableCounter_final import english_steps
    import translator
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def get_english_count(line):
    _, _, _, count = english_steps(line)
    return count

# =====================================================================
# JOINT MULTI-OPTION TRANSLATION ENGINE (API Optimized)
# =====================================================================
def generate_translation_options(english_line, db_with_counts, num_options=3):
    """
    Generates num_options unique translations using a joint feedback loop.
    This reduces API calls by 3x compared to generating options individually.
    """
    target_n = get_english_count(english_line)
    few_shot = translator.sample_few_shot(db_with_counts, target_n)
    
    examples_text = "\n".join(f"  - {line} ({cnt} syllables)" for line, cnt in few_shot)
    
    prompt = (
        f"You are a Bollywood lyricist in the style of Gulzar and Javed Akhtar.\n"
        f"Translate this English lyric line into Hindi with EXACTLY {target_n} syllables.\n"
        f"English: {english_line}\n\n"
        f"Here are real Bollywood lyric lines with a similar syllable count for style reference:\n"
        f"{examples_text}\n\n"
        f"Provide {num_options} DIFFERENT poetic Hindi translations. They must all mean roughly the same thing but use different words or phrasing.\n"
        f"Format your response EXACTLY like this:\n"
        f"1. [translation 1]\n"
        f"2. [translation 2]\n"
        f"3. [translation 3]\n\n"
        f"Rules:\n"
        f"  1. Output ONLY the Hindi lines in Devanagari script. No English, no syllable counts.\n"
        f"  2. Each translation must have EXACTLY {target_n} syllables.\n"
        f"  3. Do not repeat the same words across options if possible; provide variety.\n"
    )
    
    options = []
    
    for attempt in range(1, 4):
        response = translator.call_llm(prompt)
        
        # Parse the options from the numbered list
        parsed_options = []
        for line in response.split('\n'):
            line = line.strip()
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                hindi_text = match.group(1).strip()
                hindi_text = re.sub(r'[a-zA-Z\[\]\(\)]', '', hindi_text).strip()
                if len(hindi_text) > 0:
                    count = translator.get_hindi_syllable_count(hindi_text)
                    parsed_options.append({"hindi": hindi_text, "syllables": count})
                    
        # Pad to num_options if parsing failed to extract enough lines
        while len(parsed_options) < num_options:
            parsed_options.append({"hindi": "अनुवाद विफल", "syllables": 0})
            
        options = parsed_options[:num_options]
        
        # Check if they are all correct
        all_correct = True
        feedback_msgs = []
        for idx, opt in enumerate(options):
            actual = opt['syllables']
            if actual != target_n:
                all_correct = False
                diff = abs(actual - target_n)
                direction = "shorter" if actual > target_n else "longer"
                feedback_msgs.append(
                    f"Option {idx+1} ('{opt['hindi']}') has {actual} syllables. "
                    f"Please rewrite it to be {direction} by {diff} syllable(s) so it has exactly {target_n} syllables."
                )
                
        if all_correct:
            break
            
        # Otherwise build feedback prompt for the next attempt
        prompt += f"\n\nYour last output was:\n{response}\n"
        prompt += "\n".join(feedback_msgs)
        prompt += f"\nProvide {num_options} revised options, keeping the correct ones unchanged."
        
    return target_n, options

def process_text_file(input_txt, csv_output):
    print(f"Reading {input_txt}...")
    with open(input_txt, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and len(line.strip()) > 2]
        
    print("Loading database cache...")
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_cache.pkl")
    if not os.path.exists(cache_path):
        print("Error: db_cache.pkl not found. Run translator.py first.")
        return []
        
    with open(cache_path, 'rb') as f:
        db_with_counts = pickle.load(f)
        
    results = []
    print("\nGenerating translation options from LLM...")
    for i, eng_line in enumerate(lines):
        print(f"  Line {i+1}/{len(lines)}: {eng_line}")
        target_syl, options = generate_translation_options(eng_line, db_with_counts)
        results.append({
            "english": eng_line,
            "target_syllables": target_syl,
            "options": options
        })
        
    # Save to CSV
    with open(csv_output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = ["English_Line", "Target_Syllables"]
        for i in range(3):
            header.extend([f"Option_{i+1}_Hindi", f"Option_{i+1}_Syllables"])
        writer.writerow(header)
        
        for r in results:
            row = [r['english'], r['target_syllables']]
            for opt in r['options']:
                row.extend([opt['hindi'], opt['syllables']])
            writer.writerow(row)
            
    print(f"\nSaved translation options to {csv_output}")
    return results

def interactive_selection(results, final_txt_output):
    print("\n" + "="*50)
    print(" 🎯 INTERACTIVE TRANSLATION SELECTION 🎯")
    print("="*50)
    
    selected_lines = []
    for i, r in enumerate(results):
        print(f"\nLine {i+1}: {r['english']} (Target: {r['target_syllables']} syllables)")
        
        valid_choices = []
        for j, opt in enumerate(r['options']):
            diff = opt['syllables'] - r['target_syllables']
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            match_str = "✅ MATCH" if diff == 0 else f"❌ {diff_str}"
            print(f"  [{j+1}] {opt['hindi']} ({opt['syllables']} syl | {match_str})")
            valid_choices.append(str(j+1))
            
        print(f"  [4] Type your own custom translation")
        
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_txt = os.path.join(script_dir, "shape_of_you_lyrics.txt")
    csv_output = os.path.join(script_dir, "translation_options.csv")
    final_txt_output = os.path.join(script_dir, "final_hindi_lyrics.txt")
    audio_output = os.path.join(script_dir, "..", "Synthesize", "final_spoken_lyrics.wav")
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)
        
    if not os.path.exists(input_txt):
        print(f"Input file not found: {input_txt}")
        sys.exit(1)
        
    results = process_text_file(input_txt, csv_output)
    
    if results:
        selected = interactive_selection(results, final_txt_output)
        generate_tts_audio(selected, audio_output)
