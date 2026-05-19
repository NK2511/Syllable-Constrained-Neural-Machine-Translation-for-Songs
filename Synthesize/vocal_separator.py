import os
import sys
import subprocess

def separate_vocals(audio_file_path):
    if not os.path.exists(audio_file_path):
        print(f"Error: The file '{audio_file_path}' does not exist.")
        print("Please place your song in the Synthesize folder and try again.")
        sys.exit(1)

    print(f"Starting vocal separation for: {audio_file_path}")
    print("This will download the Demucs model (if first time) and may take several minutes depending on your CPU/GPU...")
    
    # Run Demucs via command line (installed via pip)
    # We use 'htdemucs' model and extract two stems: vocals and other (instrumental)
    try:
        command = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems=vocals",
            audio_file_path
        ]
        
        # Execute the command
        process = subprocess.run(command, check=True)
        
        print("\n✅ Separation Complete!")
        print("Check the 'separated/htdemucs/' folder in this directory for your isolated 'vocals.wav' and 'no_vocals.wav' tracks.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during separation: {e}")
        print("Please make sure you have installed demucs correctly: pip install -r requirements_synthesis.txt")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("===========================================")
    print(" 🎙️ AI Vocal Extractor (Demucs Engine) 🎙️")
    print("===========================================\n")
    
    if len(sys.argv) < 2:
        print("Usage: python vocal_separator.py <your_song_file.mp3>")
        print("Example: python vocal_separator.py shape_of_you.mp3")
    else:
        audio_path = sys.argv[1]
        separate_vocals(audio_path)
