import random
import torch
from sentence_transformers import util
from nltk.corpus import wordnet

from SyllableCounter_final import (
    count_english_syllables,
    count_hindi_syllables
)


class SynonymSwapper:
    def __init__(self, model, db_lines, db_embeddings):
        self.model = model
        self.db_lines = db_lines
        self.db_embeddings = db_embeddings

    # =====================================
    # SYNONYM GENERATION
    # =====================================
    def get_synonyms(self, word):
        synonyms = set()

        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                w = lemma.name().replace("_", " ").lower()

                if w != word.lower():
                    synonyms.add(w)

        return list(synonyms)[:3]

    # =====================================
    # GENERATE VARIANTS
    # =====================================
    def generate_variants(self, sentence):
        words = sentence.split()

        variants = [sentence]

        for i, word in enumerate(words):
            syns = self.get_synonyms(word)

            for s in syns:
                temp = words.copy()
                temp[i] = s
                variants.append(" ".join(temp))

        return list(set(variants))

    # =====================================
    # GET CANDIDATE POOL
    # =====================================
    def get_candidate_pool(self, english_line, pool_size=50):

        variants = self.generate_variants(english_line)

        all_candidates = []

        target_syl = count_english_syllables(english_line)

        for variant in variants:

            en_emb = self.model.encode(
                variant,
                convert_to_tensor=True
            )

            sims = util.pytorch_cos_sim(
                en_emb,
                self.db_embeddings
            )[0]

            top = torch.topk(
                sims,
                k=min(pool_size, len(self.db_lines))
            )

            for score, idx in zip(top[0], top[1]):

                line = self.db_lines[idx.item()]

                hi_syl = count_hindi_syllables(line)

                all_candidates.append({
                    'line': line,
                    'semantic_score': float(score),
                    'hi_syllables': hi_syl,
                    'syl_diff': abs(hi_syl - target_syl),
                    'variant': variant
                })

        # remove duplicates
        seen = set()
        unique_candidates = []

        for c in all_candidates:
            if c['line'] not in seen:
                unique_candidates.append(c)
                seen.add(c['line'])

        # prioritize syllables first
        unique_candidates.sort(
            key=lambda x: (
                x['syl_diff'],
                -x['semantic_score']
            )
        )

        return unique_candidates

    # =====================================
    # TOP K
    # =====================================
    def top_k(self, sentence, k=5):
        pool = self.get_candidate_pool(sentence)
        return pool[:k]

    # =====================================
    # BEST MATCH
    # =====================================
    def best(self, sentence):
        pool = self.get_candidate_pool(sentence)
        return pool[0] if pool else None