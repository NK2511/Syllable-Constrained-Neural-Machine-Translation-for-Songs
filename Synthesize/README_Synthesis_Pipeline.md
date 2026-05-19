# 🎤 Singing Voice Synthesis & Vocal Transfer Pipeline

To make an AI sing your translated Hindi lyrics to the exact tune of the original English song (e.g., "Shape of You"), you will use a process called **Audio-to-Audio Vocal Conversion**. 

Here is the step-by-step pipeline you need to follow. I have provided the necessary scripts in this folder to get you started.

## Step 1: Get the Original Audio
You need the original song in `.mp3` or `.wav` format. 
* Place the song file (e.g., `shape_of_you.mp3`) into this `Synthesize` folder.

## Step 2: Isolate the Vocals and Instrumental (Vocal Separation)
Before we can map new lyrics to the melody, we need to strip away the drums, bass, and instruments.
* **What you do:** Run the `vocal_separator.py` script provided in this folder. It uses **Demucs** (a state-of-the-art Meta AI deep learning model) to split the song into two files:
  1. `vocals.wav` (Just the original singer's voice)
  2. `no_vocals.wav` (The instrumental beat)

## Step 3: Record a "Guide Vocal"
## Step 3: Auto-Generate a "Guide Vocal" (No Singing Required!)
If you don't want to record yourself, we can use an AI Text-To-Speech (TTS) engine to speak the Hindi lyrics for us!
* **What you do:** Run a Python script to convert your translated Hindi lyrics into speech using Google TTS (`gTTS`). 
* Because your Hindi line has the exact same number of syllables as the English line, you just need to time-stretch (speed up or slow down) the TTS audio to match the exact length of the original isolated vocal clip.
* Save this perfectly timed TTS audio as `hindi_guide.wav`.

## Step 4: Voice Conversion (RVC) and Pitch Transfer
Now you use an AI Voice Conversion tool (like **RVC - Retrieval-based Voice Conversion** or **Diff-SVC**).
* You feed the RVC model your robotic `hindi_guide.wav`.
* You tell the RVC model to extract the musical pitch ($F_0$ contour) from the original `vocals.wav` using algorithms like **Crepe** or **RMVPE**.
* **The Result:** The AI will take the robotic Hindi pronunciation, completely replace the robotic voice with a beautiful, professional singing voice, and force the pitch to match the original singer's exact melody!

## Step 5: Mix it together
Mix your newly generated Hindi AI vocals with the original `no_vocals.wav` instrumental track using Python's `pydub` or any audio software (like FL Studio or Audacity).

---

### How to start right now:
1. Open your terminal and activate your virtual environment.
2. Install the required audio libraries by running:
   `pip install -r requirements_synthesis.txt`
3. Place your song in this folder as `song.mp3`.
4. Run the vocal separator:
   `python vocal_separator.py song.mp3`
