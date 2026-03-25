"""
Scoring Module
==============
Scores Hindi lyric candidates on three axes:
  1. Syllable Match  — does it fit the melody?
  2. Semantic Similarity — does it preserve meaning?
  3. Fluency / Naturalness — does it sound like a real Hindi song line?

Uses sentence-transformers (multilingual) for semantic similarity
and a lightweight approach for fluency scoring.
"""

import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from syllable_counter import count_hindi_syllables


# ─── Model Loading (Lazy Singleton) ─────────────────────────────────────────

_semantic_model = None
_device = None


def _get_semantic_model():
    """Lazy-load the multilingual sentence transformer."""
    global _semantic_model, _device
    if _semantic_model is None:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading semantic model on {_device}...")
        # This model maps 50+ languages to the same vector space
        _semantic_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            device=_device
        )
        print("Semantic model loaded.")
    return _semantic_model


# ─── Score 1: Syllable Match ────────────────────────────────────────────────

def score_syllable_match(
    hindi_text: str,
    target_syllables: int,
    tolerance: int = 1
) -> float:
    """
    Score how well the Hindi text matches the target syllable count.
    
    Args:
        hindi_text: Hindi text to score
        target_syllables: Target syllable count
        tolerance: Number of syllables of leeway for perfect score (default: 1)
    
    Returns:
        Score between 0.0 and 1.0
        - 1.0 = exact match or within tolerance
        - Decays smoothly for larger deviations
    """
    if target_syllables <= 0:
        return 0.0
    
    hindi_syllables = count_hindi_syllables(hindi_text)
    diff = abs(hindi_syllables - target_syllables)
    
    if diff <= tolerance:
        return 1.0
    
    # Smooth decay: penalize deviations beyond tolerance
    # Using exponential decay for smooth falloff
    overshoot = diff - tolerance
    score = np.exp(-0.3 * overshoot)
    
    return float(max(0.0, score))


# ─── Score 2: Semantic Similarity ────────────────────────────────────────────

def score_semantic_similarity(
    english_text: str,
    hindi_text: str,
) -> float:
    """
    Score semantic similarity between English source and Hindi candidate.
    Uses multilingual sentence embeddings.
    
    Args:
        english_text: Source English text
        hindi_text: Candidate Hindi text
    
    Returns:
        Cosine similarity score between 0.0 and 1.0
    """
    model = _get_semantic_model()
    
    # Encode both texts
    embeddings = model.encode(
        [english_text, hindi_text],
        convert_to_tensor=True,
        show_progress_bar=False
    )
    
    # Cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0),
        embeddings[1].unsqueeze(0)
    ).item()
    
    # Clamp to [0, 1] (cosine sim can be negative but we treat that as 0)
    return float(max(0.0, similarity))


def score_semantic_similarity_batch(
    english_text: str,
    hindi_candidates: list[str],
) -> list[float]:
    """
    Score semantic similarity for multiple Hindi candidates at once (efficient).
    
    Args:
        english_text: Source English text
        hindi_candidates: List of Hindi candidate texts
    
    Returns:
        List of similarity scores
    """
    if not hindi_candidates:
        return []
    
    model = _get_semantic_model()
    
    # Encode all texts at once
    all_texts = [english_text] + hindi_candidates
    embeddings = model.encode(
        all_texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=32
    )
    
    english_emb = embeddings[0].unsqueeze(0)
    hindi_embs = embeddings[1:]
    
    # Batch cosine similarity
    similarities = torch.nn.functional.cosine_similarity(
        english_emb, hindi_embs
    )
    
    return [float(max(0.0, s)) for s in similarities.tolist()]


# ─── Score 3: Fluency / Naturalness ─────────────────────────────────────────

def score_fluency(hindi_text: str) -> float:
    """
    Score the fluency/naturalness of Hindi text using heuristic features.
    
    This is a lightweight approach that checks for:
    - Word length distribution (natural Hindi lines have varied word lengths)
    - Absence of broken/incomplete words
    - Proper Devanagari usage
    - Line length appropriate for a song line
    
    Args:
        hindi_text: Hindi text to score
    
    Returns:
        Score between 0.0 and 1.0
    """
    if not hindi_text or not hindi_text.strip():
        return 0.0
    
    score = 1.0
    text = hindi_text.strip()
    
    # --- Feature 1: Contains Devanagari text ---
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total_chars = len(re.sub(r'\s', '', text))
    if total_chars == 0:
        return 0.0
    devanagari_ratio = devanagari_chars / total_chars
    if devanagari_ratio < 0.8:
        score *= 0.5  # Penalize mixed-script text
    
    # --- Feature 2: Reasonable line length ---
    # Song lines typically have 3-15 words
    words = text.split()
    num_words = len(words)
    if num_words < 2:
        score *= 0.4
    elif num_words > 20:
        score *= 0.5
    elif num_words > 15:
        score *= 0.7
    
    # --- Feature 3: Word length distribution ---
    # Natural Hindi has a mix of short and medium words
    if num_words > 0:
        word_lengths = [len(w) for w in words]
        avg_len = sum(word_lengths) / len(word_lengths)
        if avg_len < 1.5:
            score *= 0.6  # Too many single-char words
        elif avg_len > 8:
            score *= 0.7  # Unusually long words
    
    # --- Feature 4: No broken Unicode ---
    # Check for orphaned matras (vowel signs without preceding consonant)
    orphan_pattern = re.compile(r'(?:^|[\s])[\u093E-\u094C\u0962\u0963]')
    if orphan_pattern.search(text):
        score *= 0.3
    
    # --- Feature 5: Not just repetition ---
    if num_words >= 3:
        unique_words = set(words)
        uniqueness_ratio = len(unique_words) / num_words
        if uniqueness_ratio < 0.3:
            score *= 0.5  # Too repetitive
    
    # --- Feature 6: Common Hindi song words (bonus) ---
    # Presence of common song vocabulary is a positive signal
    song_vocab = {
        'दिल', 'इश्क़', 'प्यार', 'ज़िंदगी', 'ख्वाब', 'रात', 'चाँद',
        'तेरा', 'तेरी', 'तेरे', 'मेरा', 'मेरी', 'मेरे', 'तुम', 'हम',
        'आज', 'कल', 'नज़र', 'रोशनी', 'सितारे', 'आसमान', 'धड़कन',
        'मोहब्बत', 'जुनून', 'फ़ितूर', 'वफ़ा', 'रंग', 'सपने', 'बारिश',
        'होंठ', 'आँखें', 'बाहें', 'साथ', 'रूह', 'जान', 'ख़ुशी',
    }
    words_set = set(words)
    song_word_count = len(words_set & song_vocab)
    if song_word_count > 0:
        score = min(1.0, score * (1.0 + 0.05 * song_word_count))
    
    return float(min(1.0, max(0.0, score)))


# ─── Combined Scorer ────────────────────────────────────────────────────────

class LyricScorer:
    """
    Scores Hindi lyric candidates using a weighted combination of
    syllable match, semantic similarity, and fluency.
    """
    
    def __init__(
        self,
        w_syllable: float = 0.35,
        w_semantic: float = 0.40,
        w_fluency: float = 0.25,
        syllable_tolerance: int = 1,
    ):
        """
        Args:
            w_syllable: Weight for syllable match score
            w_semantic: Weight for semantic similarity score
            w_fluency: Weight for fluency/naturalness score
            syllable_tolerance: Syllable count tolerance (default: ±1)
        """
        total = w_syllable + w_semantic + w_fluency
        self.w_syllable = w_syllable / total
        self.w_semantic = w_semantic / total
        self.w_fluency = w_fluency / total
        self.syllable_tolerance = syllable_tolerance
    
    def score_candidates(
        self,
        english_line: str,
        hindi_candidates: list[str],
        target_syllables: int,
    ) -> list[dict]:
        """
        Score all Hindi candidates and return sorted results.
        
        Args:
            english_line: Source English lyric line
            hindi_candidates: List of Hindi candidate translations
            target_syllables: Target syllable count
        
        Returns:
            List of dicts sorted by final_score (best first), each containing:
            - hindi_text
            - syllable_count
            - syllable_score
            - semantic_score
            - fluency_score
            - final_score
        """
        if not hindi_candidates:
            return []
        
        # Batch semantic similarity (efficient)
        semantic_scores = score_semantic_similarity_batch(english_line, hindi_candidates)
        
        results = []
        for i, candidate in enumerate(hindi_candidates):
            hindi_syllables = count_hindi_syllables(candidate)
            
            syl_score = score_syllable_match(candidate, target_syllables, self.syllable_tolerance)
            sem_score = semantic_scores[i]
            flu_score = score_fluency(candidate)
            
            final_score = (
                self.w_syllable * syl_score +
                self.w_semantic * sem_score +
                self.w_fluency * flu_score
            )
            
            results.append({
                'hindi_text': candidate,
                'syllable_count': hindi_syllables,
                'target_syllables': target_syllables,
                'syllable_score': round(syl_score, 3),
                'semantic_score': round(sem_score, 3),
                'fluency_score': round(flu_score, 3),
                'final_score': round(final_score, 3),
            })
        
        # Sort by final score, descending
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results


# ─── Quick Test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    scorer = LyricScorer()
    
    english = "I'm walking on sunshine"
    target = 7  # syllables
    
    # Sample Hindi candidates (varying quality)
    candidates = [
        "मैं धूप पर चल रहा हूँ",       # literal translation
        "धूप में नाचूँ मैं आज",          # natural, songlike
        "रोशनी में झूमता मैं",           # creative adaptation
        "सूरज की किरणों में",            # formal
        "चमक रही है ज़िंदगी",           # loose but musical
    ]
    
    print(f"English: \"{english}\"  (target: {target} syllables)\n")
    
    results = scorer.score_candidates(english, candidates, target)
    
    print(f"{'Rank':<5} {'Syl':>4} {'SylSc':>6} {'SemSc':>6} {'FluSc':>6} {'TOTAL':>6}  Hindi")
    print("─" * 75)
    for rank, r in enumerate(results, 1):
        print(
            f"{rank:<5} {r['syllable_count']:>4} "
            f"{r['syllable_score']:>6.3f} {r['semantic_score']:>6.3f} "
            f"{r['fluency_score']:>6.3f} {r['final_score']:>6.3f}  {r['hindi_text']}"
        )
