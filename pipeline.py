"""
PC-NLT Pipeline
================
End-to-end pipeline for Prosody-Constrained Neural Lyric Translation.

Input:  English lyrics (full song or individual lines)
Output: Ranked Hindi adaptations with syllable-matched, natural-sounding lyrics
"""

import os
import re
import json
import time
from datetime import datetime

from syllable_counter import count_english_syllables, count_hindi_syllables
from candidate_generator import CandidateGenerator
from scoring import LyricScorer


class PCNLTPipeline:
    """
    End-to-end pipeline for prosody-constrained lyric translation.
    
    Usage:
        pipeline = PCNLTPipeline(api_key="your_gemini_key")
        result = pipeline.translate_line("I'm walking on sunshine")
        print(result['best']['hindi_text'])
    """
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-2.0-flash",
        num_candidates: int = 8,
        w_syllable: float = 0.35,
        w_semantic: float = 0.40,
        w_fluency: float = 0.25,
        syllable_tolerance: int = 1,
        temperature: float = 0.9,
    ):
        """
        Initialize the pipeline.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            model_name: Gemini model to use
            num_candidates: Number of Hindi candidates to generate per line
            w_syllable: Syllable match weight
            w_semantic: Semantic similarity weight
            w_fluency: Fluency weight
            syllable_tolerance: Acceptable syllable deviation (±)
            temperature: LLM creativity (0=conservative, 1=creative)
        """
        self.generator = CandidateGenerator(api_key=api_key, model_name=model_name)
        self.scorer = LyricScorer(
            w_syllable=w_syllable,
            w_semantic=w_semantic,
            w_fluency=w_fluency,
            syllable_tolerance=syllable_tolerance,
        )
        self.num_candidates = num_candidates
        self.temperature = temperature
    
    def translate_line(
        self,
        english_line: str,
        target_syllables: int = None,
        song_title: str = None,
        previous_hindi_lines: list[str] = None,
        song_theme: str = None,
        top_k: int = 3,
    ) -> dict:
        """
        Translate a single English lyric line to Hindi.
        
        Args:
            english_line: English lyric line
            target_syllables: Target syllable count (auto-detected if None)
            song_title: Optional song title for context
            previous_hindi_lines: Previously translated Hindi lines for continuity
            song_theme: Optional theme description
            top_k: Number of top candidates to return
        
        Returns:
            Dict with:
                - english: original English line
                - target_syllables: target count
                - best: best scoring candidate (dict)
                - top_k: top K candidates (list of dicts)
                - all_candidates: all scored candidates
        """
        # Step 1: Count syllables
        if target_syllables is None:
            target_syllables = count_english_syllables(english_line)
        
        # Step 2: Generate candidates
        candidates = self.generator.generate(
            english_line=english_line,
            num_candidates=self.num_candidates,
            target_syllables=target_syllables,
            song_title=song_title,
            previous_hindi_lines=previous_hindi_lines,
            song_theme=song_theme,
            temperature=self.temperature,
        )
        
        if not candidates:
            return {
                'english': english_line,
                'target_syllables': target_syllables,
                'best': None,
                'top_k': [],
                'all_candidates': [],
                'error': 'No candidates generated',
            }
        
        # Step 3: Score all candidates
        scored = self.scorer.score_candidates(english_line, candidates, target_syllables)
        
        return {
            'english': english_line,
            'target_syllables': target_syllables,
            'best': scored[0] if scored else None,
            'top_k': scored[:top_k],
            'all_candidates': scored,
        }
    
    def translate_song(
        self,
        lyrics: str | list[str],
        song_title: str = None,
        song_theme: str = None,
        delay_between_lines: float = 1.0,
        verbose: bool = True,
    ) -> dict:
        """
        Translate an entire song, maintaining line-by-line continuity.
        
        Args:
            lyrics: Full lyrics as string (newline-separated) or list of lines
            song_title: Song title for context
            song_theme: Theme/mood description
            delay_between_lines: Delay between API calls (rate limiting)
            verbose: Print progress
        
        Returns:
            Dict with:
                - song_title
                - lines: list of per-line results
                - hindi_lyrics: final Hindi lyrics as string
        """
        # Parse lyrics into lines
        if isinstance(lyrics, str):
            lines = [l.strip() for l in lyrics.strip().split('\n')]
        else:
            lines = [l.strip() for l in lyrics]
        
        # Filter out empty lines and section markers like [Chorus], [Verse]
        processed_lines = []
        section_markers = []
        for i, line in enumerate(lines):
            if not line:
                processed_lines.append(('empty', '', i))
            elif re.match(r'^\[.*\]$', line):
                processed_lines.append(('section', line, i))
            else:
                processed_lines.append(('lyric', line, i))
        
        # Translate each lyric line
        results = []
        translated_hindi = []
        
        for entry_type, line, orig_idx in processed_lines:
            if entry_type == 'empty':
                results.append({
                    'type': 'empty',
                    'original_index': orig_idx,
                    'english': '',
                    'hindi': '',
                })
                translated_hindi.append('')
                continue
            
            if entry_type == 'section':
                results.append({
                    'type': 'section',
                    'original_index': orig_idx,
                    'english': line,
                    'hindi': line,  # Keep section markers as-is
                })
                translated_hindi.append(line)
                continue
            
            if verbose:
                print(f"[{len(results)+1}/{len(processed_lines)}] Translating: \"{line}\"")
            
            # Get recent Hindi lines for context (last 4 non-empty)
            recent_hindi = [h for h in translated_hindi[-4:] if h and not h.startswith('[')]
            
            result = self.translate_line(
                english_line=line,
                song_title=song_title,
                previous_hindi_lines=recent_hindi if recent_hindi else None,
                song_theme=song_theme,
            )
            
            best_hindi = result['best']['hindi_text'] if result['best'] else line
            
            results.append({
                'type': 'lyric',
                'original_index': orig_idx,
                'english': line,
                'hindi': best_hindi,
                'details': result,
            })
            translated_hindi.append(best_hindi)
            
            if verbose and result['best']:
                b = result['best']
                print(f"  → {best_hindi}")
                print(f"    [syl: {b['syllable_count']}/{b['target_syllables']}  "
                      f"score: {b['final_score']:.3f}]\n")
            
            # Rate limiting
            if delay_between_lines > 0:
                time.sleep(delay_between_lines)
        
        # Assemble final Hindi lyrics
        hindi_lines = [r['hindi'] for r in results]
        hindi_lyrics = '\n'.join(hindi_lines)
        
        return {
            'song_title': song_title,
            'song_theme': song_theme,
            'lines': results,
            'hindi_lyrics': hindi_lyrics,
            'english_lyrics': '\n'.join(lines),
        }
    
    def print_comparison(self, song_result: dict):
        """Pretty-print side-by-side comparison of English and Hindi lyrics."""
        print("\n" + "═" * 80)
        if song_result.get('song_title'):
            print(f"  🎵 {song_result['song_title']}")
        print("═" * 80)
        print(f"{'ENGLISH':<40} {'HINDI':>40}")
        print("─" * 80)
        
        for entry in song_result['lines']:
            en = entry['english']
            hi = entry['hindi']
            
            if entry['type'] == 'empty':
                print()
            elif entry['type'] == 'section':
                print(f"\n  {en}")
            else:
                # Truncate long lines for display
                en_display = en[:38] if len(en) > 38 else en
                print(f"  {en_display:<38} {hi}")
        
        print("═" * 80)
    
    def save_result(self, song_result: dict, output_path: str):
        """Save translation result to JSON file."""
        # Make serializable
        output = {
            'song_title': song_result.get('song_title'),
            'song_theme': song_result.get('song_theme'),
            'timestamp': datetime.now().isoformat(),
            'english_lyrics': song_result.get('english_lyrics'),
            'hindi_lyrics': song_result.get('hindi_lyrics'),
            'lines': [],
        }
        
        for entry in song_result['lines']:
            line_data = {
                'type': entry['type'],
                'english': entry['english'],
                'hindi': entry['hindi'],
            }
            if entry['type'] == 'lyric' and 'details' in entry:
                details = entry['details']
                line_data['target_syllables'] = details['target_syllables']
                if details['best']:
                    line_data['best_score'] = details['best']['final_score']
                    line_data['best_syllable_count'] = details['best']['syllable_count']
                line_data['all_candidates'] = details.get('all_candidates', [])
            output['lines'].append(line_data)
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Result saved to: {output_path}")


# ─── CLI Interface ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PC-NLT: Prosody-Constrained Neural Lyric Translation')
    parser.add_argument('--line', type=str, help='Single English lyric line to translate')
    parser.add_argument('--file', type=str, help='Path to text file with English lyrics')
    parser.add_argument('--title', type=str, default=None, help='Song title for context')
    parser.add_argument('--theme', type=str, default=None, help='Song theme/mood')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--candidates', type=int, default=8, help='Number of candidates per line')
    parser.add_argument('--temperature', type=float, default=0.9, help='LLM creativity (0-1)')
    parser.add_argument('--api-key', type=str, default=None, help='Gemini API key')
    
    args = parser.parse_args()
    
    pipeline = PCNLTPipeline(
        api_key=args.api_key,
        num_candidates=args.candidates,
        temperature=args.temperature,
    )
    
    if args.line:
        # Single line mode
        print(f"\n🎤 English: \"{args.line}\"")
        en_syl = count_english_syllables(args.line)
        print(f"📐 Target syllables: {en_syl}\n")
        
        result = pipeline.translate_line(
            args.line,
            song_title=args.title,
            song_theme=args.theme,
        )
        
        print(f"{'Rank':<5} {'Syl':>4} {'Score':>6}  Hindi Adaptation")
        print("─" * 60)
        for rank, r in enumerate(result['all_candidates'], 1):
            marker = " ★" if rank == 1 else ""
            print(f"{rank:<5} {r['syllable_count']:>4} {r['final_score']:>6.3f}  {r['hindi_text']}{marker}")
    
    elif args.file:
        # Full song mode
        with open(args.file, 'r', encoding='utf-8') as f:
            lyrics = f.read()
        
        result = pipeline.translate_song(
            lyrics=lyrics,
            song_title=args.title,
            song_theme=args.theme,
        )
        
        pipeline.print_comparison(result)
        
        if args.output:
            pipeline.save_result(result, args.output)
    
    else:
        # Interactive mode
        print("\n🎵 PC-NLT: Prosody-Constrained Neural Lyric Translation")
        print("─" * 55)
        print("Enter English lyric lines (empty line to quit):\n")
        
        while True:
            line = input("EN > ").strip()
            if not line:
                break
            
            result = pipeline.translate_line(line)
            
            if result['best']:
                b = result['best']
                print(f"HI > {b['hindi_text']}")
                print(f"     [syllables: {b['syllable_count']}/{b['target_syllables']}  "
                      f"score: {b['final_score']:.3f}]\n")
            else:
                print("     [No candidates generated]\n")
