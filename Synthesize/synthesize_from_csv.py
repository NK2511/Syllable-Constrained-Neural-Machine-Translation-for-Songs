import os
import sys
import csv
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from gtts import gTTS
import pyworld as pw
import librosa

# Ensure we can import miniaudio
try:
    import miniaudio
except ImportError:
    print("Please install miniaudio: pip install miniaudio")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))

def auto_detect_phrases_librosa(audio, fs, top_db=22):
    """
    Robust vocal phrase detection using librosa.
    """
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=512)
    
    merged = []
    min_gap = int(0.25 * fs) # 250ms minimum silence to split phrases
    if len(intervals) > 0:
        curr_s, curr_e = intervals[0]
        for s, e in intervals[1:]:
            if s - curr_e < min_gap:
                curr_e = e # Merge close vocals
            else:
                merged.append((curr_s, curr_e))
                curr_s, curr_e = s, e
        merged.append((curr_s, curr_e))
        
    # Filter out tiny accidental noises (< 0.5 sec)
    min_len = int(0.5 * fs)
    return [(s, e) for s, e in merged if e - s > min_len]

def high_fidelity_world_vocoder(tts_speech, original_vocals, fs):
    """
    Uses the WORLD Vocoder with voicing-aware mapping to:
    1. Only stretch voiced vowels, leaving consonants unstretched to prevent distortion.
    2. Prevent unvoiced consonants (like "s") from being pitched (buzzing).
    3. Synthesize clean, human-like singing.
    """
    # 1. Analyze TTS Speech (Source)
    tts_speech = tts_speech.astype(np.float64)
    f0_source, t_source = pw.harvest(tts_speech, fs)
    sp_source = pw.cheaptrick(tts_speech, f0_source, t_source, fs)
    ap_source = pw.d4c(tts_speech, f0_source, t_source, fs)
    
    # 2. Analyze Original Vocals (Target Pitch & Timing)
    original_vocals = original_vocals.astype(np.float64)
    f0_target, t_target = pw.harvest(original_vocals, fs)
    
    target_frames = len(t_target)
    
    # 3. Voicing-Aware Timing Interpolation
    sp_stretched = np.zeros((target_frames, sp_source.shape[1]))
    ap_stretched = np.zeros((target_frames, ap_source.shape[1]))
    f0_new = np.zeros(target_frames)
    
    voiced_idx_s = np.where(f0_source > 0)[0]
    unvoiced_idx_s = np.where(f0_source == 0)[0]
    
    if len(voiced_idx_s) == 0 or len(unvoiced_idx_s) == 0:
        # Fallback to standard linear interpolation if source is entirely voiced/unvoiced
        orig_steps = np.linspace(0, 1, len(t_source))
        target_steps = np.linspace(0, 1, target_frames)
        for i in range(sp_source.shape[1]):
            sp_stretched[:, i] = np.interp(target_steps, orig_steps, sp_source[:, i])
            ap_stretched[:, i] = np.interp(target_steps, orig_steps, ap_source[:, i])
        f0_new = f0_target
    else:
        for t in range(target_frames):
            pos = t / (target_frames - 1) if target_frames > 1 else 0.0
            
            if f0_target[t] > 0:
                # Target is voiced (vowel/singing note)! Map to a voiced frame in the source
                s_idx = voiced_idx_s[int(pos * (len(voiced_idx_s) - 1))]
                sp_stretched[t, :] = sp_source[s_idx, :]
                ap_stretched[t, :] = ap_source[s_idx, :]
                f0_new[t] = f0_target[t] # Use the singing note pitch
            else:
                # Target is unvoiced (consonant/silence)! Map to an unvoiced frame in the source
                s_idx = unvoiced_idx_s[int(pos * (len(unvoiced_idx_s) - 1))]
                sp_stretched[t, :] = sp_source[s_idx, :]
                ap_stretched[t, :] = ap_source[s_idx, :]
                f0_new[t] = 0.0 # Force unvoiced consonant (no metallic buzzing!)
                
    # 4. Synthesize!
    synthesized = pw.synthesize(f0_new, sp_stretched, ap_stretched, fs, pw.default_frame_period)
    
    # Match length exactly to original vocals due to frame padding
    if len(synthesized) > len(original_vocals):
        synthesized = synthesized[:len(original_vocals)]
    elif len(synthesized) < len(original_vocals):
        synthesized = np.pad(synthesized, (0, len(original_vocals) - len(synthesized)))
        
    return synthesized

def main():
    print("==================================================")
    print(" 🎙️ High-Fidelity AI Singing Generator (WORLD) 🎙️")
    print("==================================================")

    csv_path = os.path.join(script_dir, "..", "Method_1_Retrieval", "translation_options.csv")
    vocals_path = os.path.join(script_dir, "shape_of_you_vocals.wav")
    output_vocals_path = os.path.join(script_dir, "shape_of_you_hindi_vocals.wav")

    if not os.path.exists(csv_path):
        print(f"Error: Translation options file not found at: {csv_path}")
        sys.exit(1)

    if not os.path.exists(vocals_path):
        print(f"Error: Original vocals file not found at: {vocals_path}")
        sys.exit(1)

    print("Loading original vocals file...")
    fs, original_audio = wavfile.read(vocals_path)
    if len(original_audio.shape) > 1:
        original_mono = original_audio.mean(axis=1).astype(np.float32)
    else:
        original_mono = original_audio.astype(np.float32)
    
    original_mono = original_mono / (np.max(np.abs(original_mono)) + 1e-8)
    reconstructed_vocals = np.zeros_like(original_mono)

    print("\n🔍 Auto-detecting vocal phrases with Librosa...")
    detected_phrases = auto_detect_phrases_librosa(original_mono, fs)
    print(f"✅ Found {len(detected_phrases)} vocal phrases automatically.")

    lines = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            lines.append({
                "index": idx + 1,
                "english": row["English_Line"],
                "target_syllables": int(row["Target_Syllables"]),
                "options": [row["Option_1_Hindi"], row["Option_2_Hindi"], row["Option_3_Hindi"]],
                "syllables": [int(row["Option_1_Syllables"]), int(row["Option_2_Syllables"]), int(row["Option_3_Syllables"])]
            })

    print(f"Loaded {len(lines)} lyric lines for synthesis.")
    print("--------------------------------------------------")

    for i, line in enumerate(lines):
        print(f"\nLine {line['index']}/{len(lines)}:")
        print(f"  English: \"{line['english']}\" (Target: {line['target_syllables']} syllables)")
        
        # Display options
        for o_idx in range(3):
            opt_text = line['options'][o_idx]
            opt_syl = line['syllables'][o_idx]
            diff = opt_syl - line['target_syllables']
            match_str = "✅ MATCH" if diff == 0 else f"❌ {diff:+d}"
            print(f"    [{o_idx+1}] {opt_text} ({opt_syl} syl | {match_str})")
        print("    [4] Enter custom text\n    [5] Skip this line")

        # Option selection
        selected_text = ""
        while True:
            choice = input("  Select translation (1-5): ").strip()
            if choice in ['1', '2', '3']:
                selected_text = line['options'][int(choice)-1]
                break
            elif choice == '4':
                selected_text = input("  Enter custom Hindi text: ").strip()
                break
            elif choice == '5':
                selected_text = None
                break
            else:
                print("  Invalid choice.")

        if selected_text is None:
            print("  Skipped synthesis for this line.")
            continue
            
        # Determine timing (Auto vs Manual)
        t_start_auto, t_end_auto = 0.0, 0.0
        if i < len(detected_phrases):
            t_start_auto = detected_phrases[i][0] / fs
            t_end_auto = detected_phrases[i][1] / fs
            print(f"  Auto-detected timing: {t_start_auto:.2f}s to {t_end_auto:.2f}s")
        else:
            print("  ⚠️ No auto-detected timing available for this line.")
            
        timing_input = input("  Press Enter to use Auto-timing, OR type custom timestamps (e.g., '2.5 7.3'): ").strip()
        
        if timing_input:
            try:
                t_start, t_end = map(float, timing_input.split())
            except ValueError:
                print("  Invalid input format. Skipping line.")
                continue
        else:
            if i >= len(detected_phrases):
                print("  Cannot auto-time. Skipping line.")
                continue
            t_start, t_end = t_start_auto, t_end_auto
            
        start_sample = int(t_start * fs)
        end_sample = int(t_end * fs)
        
        vocal_slice = original_mono[start_sample:end_sample]
        if len(vocal_slice) < 512:
            print("  Vocal slice too short.")
            continue

        print(f"  🎧 Processing WORLD synthesis for {t_start:.2f}s - {t_end:.2f}s...")
        
        tts = gTTS(text=selected_text, lang='hi', slow=False)
        temp_mp3 = os.path.join(script_dir, f"temp_synthesis_{line['index']}.mp3")
        tts.save(temp_mp3)
        
        try:
            decoded = miniaudio.decode_file(temp_mp3, sample_rate=fs, dither=miniaudio.DitherMode.NONE)
            speech_data = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32)
            if decoded.nchannels == 2:
                speech_data = speech_data.reshape(-1, 2).mean(axis=1)
        except Exception as e:
            print("  TTS Decode Error. Skipping.")
            continue
            
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

        # Trim silence from TTS bounds
        max_amp = np.max(np.abs(speech_data))
        if max_amp > 0:
            threshold = 0.05 * max_amp
            active_indices = np.where(np.abs(speech_data) > threshold)[0]
            if len(active_indices) > 0:
                speech_data = speech_data[active_indices[0]:active_indices[-1]]

        # Run WORLD Vocoder Synthesis
        vocoded_slice = high_fidelity_world_vocoder(speech_data, vocal_slice, fs)
        
        reconstructed_vocals[start_sample:end_sample] = vocoded_slice

    # 5. Output reconstructed file
    max_rec_amp = np.max(np.abs(reconstructed_vocals))
    if max_rec_amp > 0:
        reconstructed_vocals = reconstructed_vocals / max_rec_amp

    # Save clean vocals
    wavfile.write(output_vocals_path, fs, (reconstructed_vocals * 32767).astype(np.int16))
    print("\n" + "="*50)
    print("🎉 HIGH-FIDELITY SYNTHESIS COMPLETE!")
    print(f"Combined Hindi vocals saved to: {output_vocals_path}")
    print("="*50)

    # 6. Mix automatically with instrumental
    instrumental_path = os.path.join(script_dir, "shape_of_you_no_vocals.wav")
    output_remix_path = os.path.join(script_dir, "shape_of_you_HINDI_REMIX.wav")

    if os.path.exists(instrumental_path):
        print("\n🎧 Automatically mixing Hindi vocals with instrumental beat...")
        try:
            fs_inst, inst_audio = wavfile.read(instrumental_path)
            
            # Convert to float32
            if inst_audio.dtype == np.int16:
                inst_audio_f = inst_audio.astype(np.float32) / 32768.0
            elif inst_audio.dtype == np.int32:
                inst_audio_f = inst_audio.astype(np.float32) / 2147483648.0
            else:
                inst_audio_f = inst_audio.astype(np.float32)
                
            voc_audio_f = reconstructed_vocals.astype(np.float32)

            # Convert vocals to stereo if instrumental is stereo
            if len(inst_audio_f.shape) == 2 and len(voc_audio_f.shape) == 1:
                voc_audio_f = np.column_stack((voc_audio_f, voc_audio_f))

            # Pad/match length
            max_len = max(len(inst_audio_f), len(voc_audio_f))
            if len(inst_audio_f) < max_len:
                if len(inst_audio_f.shape) == 2:
                    inst_audio_f = np.pad(inst_audio_f, ((0, max_len - len(inst_audio_f)), (0, 0)))
                else:
                    inst_audio_f = np.pad(inst_audio_f, (0, max_len - len(inst_audio_f)))
            if len(voc_audio_f) < max_len:
                if len(voc_audio_f.shape) == 2:
                    voc_audio_f = np.pad(voc_audio_f, ((0, max_len - len(voc_audio_f)), (0, 0)))
                else:
                    voc_audio_f = np.pad(voc_audio_f, (0, max_len - len(voc_audio_f)))

            # Mix (Vocal boosted to 3.5x, Inst ducked to 0.85x)
            mixed = (inst_audio_f * 0.85) + (voc_audio_f * 3.5)

            # Normalize to prevent clipping
            max_amp = np.max(np.abs(mixed))
            if max_amp > 0.95:
                mixed = (mixed / max_amp) * 0.95

            wavfile.write(output_remix_path, fs_inst, (mixed * 32767).astype(np.int16))
            print(f"🎉 Remix successfully saved to: {output_remix_path}")
        except Exception as e:
            print(f"⚠️ Error mixing audio: {e}")
    else:
        print(f"⚠️ Instrumental not found at: {instrumental_path}. Skipping automatic mix.")

if __name__ == "__main__":
    main()
