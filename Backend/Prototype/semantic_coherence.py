from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from lexical_complexity_analyzer import LexicalComplexityAnalyzer
from liwc_analyzer import LIWCAnalyzer


class SemanticCoherenceAnalyzer:
    """Analyze semantic coherence of text using sentence embeddings and integrate with other analysis methods."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            # Force CPU device to avoid meta tensor issues
            self.model = SentenceTransformer(model_name, device='cpu')
            # Ensure model is in evaluation mode
            self.model.eval()
        except Exception as e:
            print(f"Error initializing SentenceTransformer: {e}")
            self.model = None
        self.lexical_analyzer = LexicalComplexityAnalyzer()
        self.liwc_analyzer = LIWCAnalyzer()


    def _calculate_semantic_similarity(self, text: str) -> float:
        """Calculate semantic similarity between sentences in the text."""
        if not self.model:
            return 0.0
            
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.0

        try:
            embeddings = self.model.encode(sentences)
            similarities = []
            for i in range(len(sentences) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(sim)
            return float(np.mean(similarities)) if similarities else 0.0
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text for semantic coherence and other metrics."""
        if not text:
            return {}

        try:
            # Get semantic coherence
            semantic_coherence = self._calculate_semantic_similarity(text)

            # Get lexical complexity
            lexical_metrics = self.lexical_analyzer.get_stats(text)

            # Get LIWC analysis
            liwc_metrics = self.liwc_analyzer.analyze_text(text)

            # Combine results (no model predictions here)
            results = {
                "semantic_coherence": semantic_coherence,
                **lexical_metrics,
                **liwc_metrics
            }
            return results
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            return {
                "semantic_coherence": 0.0,

                **lexical_metrics,
                **liwc_metrics
            }
