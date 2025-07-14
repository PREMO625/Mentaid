from typing import Dict

import spacy


class POSUsageAnalyzer:
    """Analyse pronoun person and verb tense usage using spaCy POS tags."""

    FIRST_PRON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    SECOND_PRON = {"you", "your", "yours"}

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            self.nlp = spacy.load(model_name, disable=["ner", "textcat"])
        except Exception:
            self.nlp = None
            print("[POSUsageAnalyzer] spaCy model not found – analysis disabled.")

    def get_stats(self, text: str) -> Dict[str, Dict[str, float]]:
        if not text or not self.nlp:
            return {}

        doc = self.nlp(text)
        pronoun_counts = {"first_person": 0, "second_person": 0, "third_person": 0}
        tense_counts = {"past": 0, "present": 0}

        for tok in doc:
            # Pronouns
            if tok.pos_ == "PRON":
                lower = tok.text.lower()
                if lower in self.FIRST_PRON:
                    pronoun_counts["first_person"] += 1
                elif lower in self.SECOND_PRON:
                    pronoun_counts["second_person"] += 1
                else:
                    pronoun_counts["third_person"] += 1

            # Verb tenses (simple heuristic using morphological features)
            if tok.pos_ == "VERB":
                if "Tense=Past" in tok.morph:
                    tense_counts["past"] += 1
                else:
                    tense_counts["present"] += 1

        total_pron = sum(pronoun_counts.values()) or 1
        total_tense = sum(tense_counts.values()) or 1

        # Convert to proportion
        pronoun_props = {k: v / total_pron for k, v in pronoun_counts.items()}
        tense_props = {k: v / total_tense for k, v in tense_counts.items()}

        return {"pronouns": pronoun_props, "tenses": tense_props}
