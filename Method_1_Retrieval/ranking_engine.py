"""
Ranking Engine
==============
Implements the 4 metrics from metrics_definition.md.

  1. Purport        - high-level vibe / emotional intent (cosine sim)
  2. Meaning        - literal semantic accuracy (cosine sim, same model)
  3. Syllable Match - rhythmic constraint (exact count ratio)
  4. Bollywood Prob - how "song-like" the Hindi line is (word frequency in DB)
"""

import re
from collections import Counter

import torch
from sentence_transformers import util

from syllable_counter import count_english_syllables, count_hindi_syllables


class RankingEngine:
    def __init__(self, model, db_lines: list):
        self.model = model
        self._word_freq = self._build_word_freq(db_lines)

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_word_freq(db_lines: list) -> Counter:
        words = []
        for line in db_lines:
            words.extend(re.findall(r'[\u0900-\u097F]+', line))
        return Counter(words)

    def _embed(self, text: str):
        return self.model.encode(text, convert_to_tensor=True)

    # ── individual metrics ────────────────────────────────────────────────────

    def score_purport(self, en_emb, hindi_line: str) -> float:
        hi_emb = self._embed(hindi_line)
        return float(util.pytorch_cos_sim(en_emb, hi_emb)[0][0])

    def score_meaning(self, en_emb, hindi_line: str) -> float:
        # Same model for now; can be swapped for a dedicated MT model later
        return self.score_purport(en_emb, hindi_line)

    @staticmethod
    def score_syllable_match(english_line: str, hindi_line: str) -> float:
        en_syl = count_english_syllables(english_line)
        hi_syl = count_hindi_syllables(hindi_line)
        if en_syl == 0:
            return 1.0
        diff = abs(en_syl - hi_syl)
        return max(0.0, 1.0 - diff / en_syl)

    def score_bollywood_prob(self, hindi_line: str) -> float:
        words = re.findall(r'[\u0900-\u097F]+', hindi_line)
        if not words:
            return 0.0
        hits = sum(1 for w in words if w in self._word_freq)
        return hits / len(words)

    # ── combined score ────────────────────────────────────────────────────────

    def score_all(self, english_line: str, hindi_line: str,
                  weights: dict, en_emb=None) -> dict:
        """
        Returns a dict with all 4 metrics + final weighted score.
        weights: {'purport': float, 'meaning': float, 'syllable': float, 'bollywood': float}
        """
        if en_emb is None:
            en_emb = self._embed(english_line)

        purport  = self.score_purport(en_emb, hindi_line)
        meaning  = self.score_meaning(en_emb, hindi_line)
        syllable = self.score_syllable_match(english_line, hindi_line)
        bollywood = self.score_bollywood_prob(hindi_line)

        # Normalise weights so they always sum to 1
        total_w = sum(weights.values()) or 1.0
        w = {k: v / total_w for k, v in weights.items()}

        final = (w['purport']   * purport  +
                 w['meaning']   * meaning  +
                 w['syllable']  * syllable +
                 w['bollywood'] * bollywood)

        return {
            'purport':      purport,
            'meaning':      meaning,
            'syllable':     syllable,
            'bollywood':    bollywood,
            'final':        final,
            'en_syllables': count_english_syllables(english_line),
            'hi_syllables': count_hindi_syllables(hindi_line),
        }
