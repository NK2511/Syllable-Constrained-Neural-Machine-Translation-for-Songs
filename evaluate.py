"""
Evaluation Module
=================
Evaluates PC-NLT output quality using standard NLP metrics:
- BLEU (n-gram overlap)
- BERTScore (semantic similarity using BERT)
- Syllable Match Rate
- SacreBLEU

Also provides aggregate statistics and per-line breakdowns.
"""

import re
import numpy as np
from collections import defaultdict

from syllable_counter import count_english_syllables, count_hindi_syllables


# ─── Metric: Syllable Match ────────────────────────────────────────────────

def evaluate_syllable_match(results: list[dict], tolerance: int = 1) -> dict:
    """
    Evaluate syllable matching across all translated lines.
    
    Args:
        results: List of line results from pipeline.translate_song()
        tolerance: Acceptable deviation
    
    Returns:
        Dict with match statistics
    """
    lyric_lines = [r for r in results if r.get('type') == 'lyric']
    
    if not lyric_lines:
        return {'error': 'No lyric lines found'}
    
    exact_matches = 0
    within_tolerance = 0
    total_deviation = 0
    deviations = []
    
    for line in lyric_lines:
        en_syl = count_english_syllables(line['english'])
        hi_syl = count_hindi_syllables(line['hindi'])
        
        dev = abs(en_syl - hi_syl)
        deviations.append(dev)
        total_deviation += dev
        
        if dev == 0:
            exact_matches += 1
        if dev <= tolerance:
            within_tolerance += 1
    
    n = len(lyric_lines)
    return {
        'total_lines': n,
        'exact_match_count': exact_matches,
        'exact_match_rate': round(exact_matches / n, 3),
        'within_tolerance_count': within_tolerance,
        'within_tolerance_rate': round(within_tolerance / n, 3),
        'tolerance': tolerance,
        'mean_deviation': round(total_deviation / n, 2),
        'median_deviation': round(float(np.median(deviations)), 2),
        'max_deviation': max(deviations),
        'deviations': deviations,
    }


# ─── Metric: Semantic Similarity (BERTScore) ───────────────────────────────

def evaluate_bertscore(
    english_lines: list[str],
    hindi_lines: list[str],
    lang: str = 'hi',
) -> dict:
    """
    Compute BERTScore between English sources and Hindi translations.
    
    Args:
        english_lines: List of English source lines
        hindi_lines: List of Hindi translation lines
        lang: Language code for BERTScore
    
    Returns:
        Dict with precision, recall, F1 scores
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        return {'error': 'bert-score not installed. Run: pip install bert-score'}
    
    # BERTScore expects same-length lists
    assert len(english_lines) == len(hindi_lines), "Mismatched line counts"
    
    P, R, F1 = bert_score(
        hindi_lines,    # candidates
        english_lines,  # references
        lang=lang,
        verbose=False,
    )
    
    return {
        'precision': round(P.mean().item(), 4),
        'recall': round(R.mean().item(), 4),
        'f1': round(F1.mean().item(), 4),
        'per_line_f1': [round(f.item(), 4) for f in F1],
    }


# ─── Metric: BLEU Score ────────────────────────────────────────────────────

def evaluate_bleu(
    reference_lines: list[str],
    candidate_lines: list[str],
) -> dict:
    """
    Compute corpus-level BLEU score using SacreBLEU.
    
    Note: BLEU requires reference translations. If you don't have human 
    references, use BERTScore instead (cross-lingual).
    
    Args:
        reference_lines: Human reference Hindi translations
        candidate_lines: PC-NLT Hindi outputs
    
    Returns:
        Dict with BLEU score and details
    """
    try:
        import sacrebleu
    except ImportError:
        return {'error': 'sacrebleu not installed. Run: pip install sacrebleu'}
    
    # SacreBLEU expects list of references (can have multiple refs)
    refs = [reference_lines]
    
    bleu = sacrebleu.corpus_bleu(candidate_lines, refs)
    
    return {
        'bleu_score': round(bleu.score, 2),
        'bleu_details': str(bleu),
    }


# ─── Full Evaluation Report ────────────────────────────────────────────────

def evaluate_song_result(
    song_result: dict,
    reference_hindi: list[str] = None,
    tolerance: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Run full evaluation on a song translation result.
    
    Args:
        song_result: Output from PCNLTPipeline.translate_song()
        reference_hindi: Optional human reference translations for BLEU
        tolerance: Syllable tolerance for matching
        verbose: Print report
    
    Returns:
        Dict with all evaluation metrics
    """
    lines = song_result.get('lines', [])
    lyric_lines = [l for l in lines if l.get('type') == 'lyric']
    
    english = [l['english'] for l in lyric_lines]
    hindi = [l['hindi'] for l in lyric_lines]
    
    report = {
        'song_title': song_result.get('song_title', 'Unknown'),
        'num_lines': len(lyric_lines),
    }
    
    # 1. Syllable Match
    syl_eval = evaluate_syllable_match(lyric_lines, tolerance)
    report['syllable_match'] = syl_eval
    
    # 2. BERTScore (cross-lingual semantic similarity)
    if len(english) > 0 and len(hindi) > 0:
        bert_eval = evaluate_bertscore(english, hindi)
        report['bertscore'] = bert_eval
    
    # 3. BLEU (only if references provided)
    if reference_hindi and len(reference_hindi) == len(hindi):
        bleu_eval = evaluate_bleu(reference_hindi, hindi)
        report['bleu'] = bleu_eval
    
    # 4. Per-line breakdown
    per_line = []
    for i, ll in enumerate(lyric_lines):
        en_syl = count_english_syllables(ll['english'])
        hi_syl = count_hindi_syllables(ll['hindi'])
        entry = {
            'english': ll['english'],
            'hindi': ll['hindi'],
            'en_syllables': en_syl,
            'hi_syllables': hi_syl,
            'syllable_diff': hi_syl - en_syl,
        }
        if 'details' in ll and ll['details'].get('best'):
            entry['score'] = ll['details']['best']['final_score']
        per_line.append(entry)
    report['per_line'] = per_line
    
    # Print report
    if verbose:
        _print_report(report)
    
    return report


def _print_report(report: dict):
    """Pretty-print evaluation report."""
    print("\n" + "═" * 70)
    print(f"  📊 EVALUATION REPORT: {report.get('song_title', 'Unknown')}")
    print("═" * 70)
    
    # Syllable stats
    syl = report.get('syllable_match', {})
    if syl and 'error' not in syl:
        print(f"\n  🎵 Syllable Matching ({syl['total_lines']} lines)")
        print(f"     Exact match rate:     {syl['exact_match_rate']:.1%}")
        print(f"     Within ±{syl['tolerance']} tolerance:  {syl['within_tolerance_rate']:.1%}")
        print(f"     Mean deviation:       {syl['mean_deviation']:.1f} syllables")
        print(f"     Max deviation:        {syl['max_deviation']} syllables")
    
    # BERTScore
    bert = report.get('bertscore', {})
    if bert and 'error' not in bert:
        print(f"\n  🧠 Semantic Similarity (BERTScore)")
        print(f"     Precision: {bert['precision']:.4f}")
        print(f"     Recall:    {bert['recall']:.4f}")
        print(f"     F1:        {bert['f1']:.4f}")
    
    # BLEU
    bleu = report.get('bleu', {})
    if bleu and 'error' not in bleu:
        print(f"\n  📝 BLEU Score")
        print(f"     Score: {bleu['bleu_score']:.2f}")
    
    # Per-line summary
    per_line = report.get('per_line', [])
    if per_line:
        print(f"\n  📋 Per-Line Breakdown")
        print(f"     {'#':<3} {'EnSyl':>5} {'HiSyl':>5} {'Diff':>5} {'Score':>6}  English → Hindi")
        print("     " + "─" * 65)
        for i, pl in enumerate(per_line, 1):
            score_str = f"{pl.get('score', 0):.3f}" if 'score' in pl else "  N/A"
            diff = pl['syllable_diff']
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            en_short = pl['english'][:25] + "..." if len(pl['english']) > 25 else pl['english']
            hi_short = pl['hindi'][:25] + "..." if len(pl['hindi']) > 25 else pl['hindi']
            print(f"     {i:<3} {pl['en_syllables']:>5} {pl['hi_syllables']:>5} "
                  f"{diff_str:>5} {score_str:>6}  {en_short} → {hi_short}")
    
    print("\n" + "═" * 70)


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test with mock data
    mock_result = {
        'song_title': 'Test Song',
        'lines': [
            {
                'type': 'lyric',
                'english': "I'm walking on sunshine",
                'hindi': "धूप में नाचूँ मैं आज",
                'details': {'best': {'final_score': 0.85, 'syllable_count': 7, 'target_syllables': 7}},
            },
            {
                'type': 'lyric', 
                'english': "And don't it feel good",
                'hindi': "कितना अच्छा लगता है",
                'details': {'best': {'final_score': 0.78, 'syllable_count': 8, 'target_syllables': 5}},
            },
        ],
    }
    
    evaluate_song_result(mock_result)
