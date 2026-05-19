"""
Quick TTS diagnostic — run this and paste the output.
Usage: nlp_venv/Scripts/python.exe test_tts.py
"""

import pythoncom
import win32com.client

pythoncom.CoInitialize()

speaker = win32com.client.Dispatch('SAPI.SpVoice')

print("=== Available voices ===")
voices = speaker.GetVoices()
for i in range(voices.Count):
    v = voices.Item(i)
    print(f"  [{i}] {v.GetDescription()}")

print(f"\nCurrent voice: {speaker.Voice.GetDescription()}")
print("Current rate:", speaker.Rate)
print("Current volume:", speaker.Volume)

print("\nSpeaking 'hello' in English...")
speaker.Speak("hello, this is a test")

print("Speaking Hindi text...")
speaker.Speak("नमस्ते, यह एक परीक्षण है")

print("\nDone. Did you hear anything?")
pythoncom.CoUninitialize()
