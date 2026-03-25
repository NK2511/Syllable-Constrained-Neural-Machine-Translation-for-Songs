"""
Candidate Generator Module
===========================
Uses Google Gemini API to generate multiple Hindi lyric adaptations
for a given English lyric line.

The prompt is carefully engineered to produce modern, Bollywood-style
Hindi lyrics — not formal translations.
"""

import os
import re
import json
import google.generativeai as genai
from syllable_counter import count_english_syllables


def _build_prompt(english_line: str, target_syllables: int, num_candidates: int = 8) -> str:
    """
    Build a carefully engineered prompt for modern Hindi lyric adaptation.
    """
    return f"""You are a celebrated modern Hindi songwriter — think Irshad Kamil, Amitabh Bhattacharya, Jaideep Sahni, or Kumaar. You write lyrics for artists like Arijit Singh, AP Dhillon, and Shreya Ghoshal.

YOUR TASK: Adapt the following English lyric line into Hindi. This is NOT a translation job — it's a CREATIVE ADAPTATION for a Hindi song.

RULES:
1. Capture the EMOTION, MOOD, and MEANING — NOT a word-for-word translation
2. Use modern, conversational Hindi/Hindustani — the kind heard in contemporary Bollywood and Indie music
3. Freely use Urdu-origin words: dil, ishq, junoon, fitoor, arzoo, roshni, sitaare, khwab, etc.
4. Target approximately {target_syllables} syllables (akshar in Devanagari) — this is CRITICAL for singability
5. Each line must feel RHYTHMIC and MUSICAL when spoken aloud — imagine singing it
6. Avoid overly formal/Sanskritized Hindi (no "प्रकाश" when "रोशनी" works better)
7. Avoid English words mixed in (no Hinglish) — keep it pure Hindi/Urdu
8. Write in Devanagari script only

English line: "{english_line}"
Target syllable count: {target_syllables} (±1 akshar tolerance)

Generate exactly {num_candidates} DIFFERENT Hindi adaptations. Each should take a slightly different creative angle while preserving the core emotion.

IMPORTANT: Return ONLY a JSON array of strings, nothing else. Example format:
["हिंदी पंक्ति एक", "हिंदी पंक्ति दो", "हिंदी पंक्ति तीन"]

Your {num_candidates} Hindi adaptations:"""


def _build_context_prompt(
    english_line: str,
    target_syllables: int,
    num_candidates: int = 8,
    song_title: str = None,
    previous_lines: list[str] = None,
    song_theme: str = None
) -> str:
    """
    Build a context-aware prompt that considers surrounding lyrics.
    """
    context_block = ""
    if song_title:
        context_block += f"\nSong Title: \"{song_title}\""
    if song_theme:
        context_block += f"\nSong Theme/Mood: {song_theme}"
    if previous_lines:
        prev = "\n".join([f"  {l}" for l in previous_lines[-4:]])  # last 4 lines for context
        context_block += f"\nPreceding Hindi lines (for continuity):\n{prev}"

    return f"""You are a celebrated modern Hindi songwriter — think Irshad Kamil, Amitabh Bhattacharya, Jaideep Sahni, or Kumaar. You write lyrics for artists like Arijit Singh, AP Dhillon, and Shreya Ghoshal.

YOUR TASK: Adapt the following English lyric line into Hindi for a song.
{context_block}

RULES:
1. Capture the EMOTION, MOOD, and MEANING — NOT a word-for-word translation
2. Use modern, conversational Hindi/Hindustani — contemporary Bollywood & Indie music style
3. Freely use Urdu-origin words: dil, ishq, junoon, fitoor, arzoo, roshni, sitaare, khwab, etc.
4. Target approximately {target_syllables} syllables (akshar in Devanagari) — CRITICAL for singability
5. Each line must feel RHYTHMIC and MUSICAL when spoken aloud
6. Maintain THEMATIC CONTINUITY with the previous lines if provided
7. Avoid overly formal Hindi — keep it natural and songlike
8. Write in Devanagari script ONLY (no English/Hinglish)

English line: "{english_line}"
Target syllable count: {target_syllables} (±1 akshar tolerance)

Generate exactly {num_candidates} DIFFERENT Hindi adaptations. Each should explore a different creative angle while maintaining the core emotion and flow.

IMPORTANT: Return ONLY a JSON array of strings, nothing else. Example:
["हिंदी पंक्ति एक", "हिंदी पंक्ति दो"]

Your {num_candidates} Hindi adaptations:"""


class CandidateGenerator:
    """
    Generates multiple Hindi lyric candidates for an English line using Gemini API.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the generator.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Gemini model to use. Default: gemini-2.0-flash (fast + good quality)
        """
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def generate(
        self,
        english_line: str,
        num_candidates: int = 8,
        target_syllables: int = None,
        song_title: str = None,
        previous_hindi_lines: list[str] = None,
        song_theme: str = None,
        temperature: float = 0.9,
    ) -> list[str]:
        """
        Generate multiple Hindi lyric candidates for an English line.
        
        Args:
            english_line: The English lyric line to adapt
            num_candidates: Number of candidates to generate (default: 8)
            target_syllables: Target syllable count. If None, auto-detected from English.
            song_title: Optional song title for context
            previous_hindi_lines: Optional list of previously translated lines for continuity
            song_theme: Optional theme/mood description
            temperature: Creativity level (0=conservative, 1=creative). Default: 0.9
        
        Returns:
            List of Hindi lyric line candidates
        """
        # Auto-detect target syllables if not specified
        if target_syllables is None:
            target_syllables = count_english_syllables(english_line)
        
        # Use context-aware prompt if we have context, otherwise simple prompt
        if song_title or previous_hindi_lines or song_theme:
            prompt = _build_context_prompt(
                english_line, target_syllables, num_candidates,
                song_title, previous_hindi_lines, song_theme
            )
        else:
            prompt = _build_prompt(english_line, target_syllables, num_candidates)
        
        # Generate with Gemini
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1024,
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        # Parse the response
        candidates = self._parse_response(response.text)
        
        # If parsing fails or returns too few, retry once
        if len(candidates) < 2:
            response = self.model.generate_content(
                prompt + "\n\nRemember: return ONLY a valid JSON array of Hindi strings.",
                generation_config=generation_config,
            )
            candidates = self._parse_response(response.text)
        
        return candidates
    
    def _parse_response(self, response_text: str) -> list[str]:
        """Parse the LLM response to extract Hindi candidates."""
        text = response_text.strip()
        
        # Try direct JSON parse
        try:
            # Remove markdown code blocks if present
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            text = text.strip()
            
            result = json.loads(text)
            if isinstance(result, list):
                return [s.strip() for s in result if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to find JSON array in the response
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [s.strip() for s in result if isinstance(s, str) and s.strip()]
            except json.JSONDecodeError:
                pass
        
        # Last resort: split by newlines and filter for Devanagari text
        lines = text.split('\n')
        candidates = []
        for line in lines:
            # Remove numbering, bullets, quotes
            cleaned = re.sub(r'^[\d\.\)\-\*\s"]+', '', line).strip().strip('"').strip()
            # Check if it contains Devanagari
            if re.search(r'[\u0900-\u097F]', cleaned) and len(cleaned) > 3:
                candidates.append(cleaned)
        
        return candidates


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        print("  set GEMINI_API_KEY=your_key_here   (Windows)")
        print("  export GEMINI_API_KEY=your_key_here (Linux/Mac)")
        sys.exit(1)
    
    gen = CandidateGenerator(api_key=api_key)
    
    test_line = "I'm walking on sunshine"
    target = count_english_syllables(test_line)
    
    print(f"English: \"{test_line}\"")
    print(f"Target syllables: {target}")
    print(f"\nGenerating Hindi candidates...\n")
    
    candidates = gen.generate(test_line)
    
    from syllable_counter import count_hindi_syllables
    for i, c in enumerate(candidates, 1):
        hs = count_hindi_syllables(c)
        match = "✓" if abs(hs - target) <= 1 else "✗"
        print(f"  {i}. [{hs:2d} akshar] {match}  {c}")
