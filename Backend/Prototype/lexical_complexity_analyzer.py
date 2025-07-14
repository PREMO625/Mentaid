import re
from typing import Dict

import spacy


class LexicalComplexityAnalyzer:
    """Compute basic lexical-complexity metrics for a text."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            self.nlp = spacy.load(model_name, disable=["ner", "textcat"])
        except Exception:
            self.nlp = None
            print("[LexicalComplexityAnalyzer] spaCy model not found – falling back to simple tokenisation.")

    def _simple_tokens(self, text: str):
        return re.findall(r"[A-Za-z']+", text.lower())

    def get_stats(self, text: str) -> Dict[str, float]:
        """Return a dictionary with lexical diversity, density, etc."""
        if not text:
            return {}

        if self.nlp:
            doc = self.nlp(text)
            tokens = [t for t in doc if not t.is_punct and not t.is_space]
            token_texts = [t.text.lower() for t in tokens]
        else:
            token_texts = self._simple_tokens(text)
            tokens = token_texts  # type: ignore

        total_tokens = len(token_texts)
        unique_tokens = len(set(token_texts))
        ttr = unique_tokens / total_tokens if total_tokens else 0.0

        # Lexical density – proportion of content words (NOUN/VERB/ADJ/ADV)
        if self.nlp:
            content_words = [t for t in tokens if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
            lexical_density = len(content_words) / total_tokens if total_tokens else 0.0
            sent_lengths = [len([t for t in s if not t.is_punct and not t.is_space]) for s in doc.sents]
        else:
            lexical_density = 0.0
            sent_lengths = []

        avg_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0.0

        return {
            "token_count": total_tokens,
            "unique_tokens": unique_tokens,
            "type_token_ratio": ttr,
            "lexical_density": lexical_density,
            "avg_sentence_length": avg_sent_len,
        }
