import os
import shutil

# Root directory of workspace
workspace_dir = os.path.dirname(os.path.abspath(__file__))
redundant_dir = os.path.join(workspace_dir, "redundant")

if not os.path.exists(redundant_dir):
    os.makedirs(redundant_dir)
    print(f"Created directory: {redundant_dir}")

# List of files to move (relative to workspace root)
files_to_move = [
    # Root level
    ("hindi_lyrics_embeddings.pt", "hindi_lyrics_embeddings.pt"),
    ("espeak-ng.msi", "espeak-ng.msi"),
    # Synthesize level
    ("Synthesize/generate_tts_guide.py", "generate_tts_guide.py"),
    ("Synthesize/sing_demo.py", "sing_demo.py"),
    ("Synthesize/mix_audio.py", "mix_audio.py"),
    # Method 3 level
    ("Method_3_LLM/list_models.py", "list_models.py"),
    # Method 1 level
    ("Method_1_Retrieval/test_tts.py", "test_tts.py"),
    ("Method_1_Retrieval/translator_gui.py", "translator_gui.py"),
    ("Method_1_Retrieval/run_translator.bat", "run_translator.bat"),
    ("Method_1_Retrieval/ranking_engine.py", "ranking_engine.py"),
    ("Method_1_Retrieval/synonym_swapper.py", "synonym_swapper.py"),
    ("Method_1_Retrieval/syllable_splitter.py", "syllable_splitter.py"),
    ("Method_1_Retrieval/syllable_counter.py", "syllable_counter.py"),
    ("Method_1_Retrieval/lyric.py", "lyric.py"),
]

for relative_src, dest_filename in files_to_move:
    src_path = os.path.join(workspace_dir, relative_src)
    dest_path = os.path.join(redundant_dir, dest_filename)
    
    if os.path.exists(src_path):
        try:
            shutil.move(src_path, dest_path)
            print(f"Moved: {relative_src} -> redundant/{dest_filename}")
        except Exception as e:
            print(f"Error moving {relative_src}: {e}")
    else:
        # Silently skip if it doesn't exist
        pass

print("\nCleanup complete!")
