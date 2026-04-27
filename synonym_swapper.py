"""
Synonym Swapper
===============
Retrieves a pool of semantically similar Hindi candidates, then selects the
best one (or a random alternative) by syllable proximity.

No external LLM needed — it leans on the existing lyrics database and the
sentence-transformer embeddings already cached by semantic_translator.py.
"""

import random
import torch
from sentence_transformers import util
from syllable_counter import count_english_syllables, count_hindi_syllables


class SynonymSwapper:
    def __init__(self, model, db_lines: list, db_embeddings):
        self.model       = model
        self.db_lines    = db_lines
        self.db_embeddings = db_embeddings

    def get_candidate_pool(self, english_line: str, pool_size: int = 60) -> list:
        """
        Return up to `pool_size` candidates, sorted by:
          1. Syllable distance (ascending)
          2. Semantic similarity (descending)
        """
        en_emb = self.model.encode(english_line, convert_to_tensor=True)
        sims   = util.pytorch_cos_sim(en_emb, self.db_embeddings)[0]
        top    = torch.topk(sims, k=min(pool_size, len(self.db_lines)))

        target_syl = count_english_syllables(english_line)
        pool = []
        for score, idx in zip(top[0], top[1]):
            line   = self.db_lines[idx.item()]
            hi_syl = count_hindi_syllables(line)
            pool.append({
                'line':          line,
                'semantic_score': float(score),
                'hi_syllables':  hi_syl,
                'syl_diff':      abs(hi_syl - target_syl),
                'en_emb':        en_emb,
            })

        pool.sort(key=lambda x: -x['semantic_score'])
        return pool

    def best(self, english_line: str, pool: list = None) -> dict:
        """Return the highest-ranked candidate (best syllable + semantic match)."""
        if pool is None:
            pool = self.get_candidate_pool(english_line)
        return pool[0] if pool else None

    def random_pick(self, pool: list, exclude_line: str = None) -> dict:
        """
        Return a random candidate from the pool, optionally excluding the
        currently shown translation so the user always gets something different.
        """
        choices = [c for c in pool if c['line'] != exclude_line]
        if not choices:
            choices = pool
        return random.choice(choices) if choices else None
