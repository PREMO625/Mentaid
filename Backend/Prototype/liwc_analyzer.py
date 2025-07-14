from empath import Empath
import spacy
from typing import Dict, List, Tuple
import re
from collections import defaultdict

class LIWCAnalyzer:
    def __init__(self):
        """
        Initialize LIWC-like analyzer with Empath and Spacy.
        """
        # Initialize Empath
        self.lexicon = Empath()
        
        # Initialize Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("Warning: Spacy model not found. Some features may be limited.")
        
        # Define desired LIWC-like categories (some may not exist in Empath)
        desired_categories = [
            "positive_emotion", "negative_emotion", "anger", "sadness",
            "joy", "fear", "surprise", "disgust", "anticipation", "trust",
            "analytic", "clout", "authentic", "tone",
            "work", "leisure", "home", "money", "religion",
            "death", "body", "health", "sexual", "ingestion",
        ]

        # Filter categories to those actually present in Empath to avoid KeyError
        empath_available = set(self.lexicon.cats)
        self.empath_categories = [cat for cat in desired_categories if cat in empath_available]
        # Keep full list to return consistent keys; unsupported ones will be filled with 0 later
        self.all_categories = desired_categories
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text using Spacy if available."""
        if self.nlp:
            doc = self.nlp(text)
            # Remove stop words and punctuation
            tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
            return " ".join(tokens)
        else:
            # Fallback preprocessing
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            return text
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text using Empath lexicon.
        
        Returns:
            Dictionary with category scores
        """
        if not text:
            return {}
            
        processed_text = self.preprocess_text(text)
        
        try:
            # Get Empath analysis
            empath_scores = self.lexicon.analyze(processed_text, categories=self.empath_categories, normalize=True)
            
            # Enhance with additional metrics
            analysis = {
                **{cat: empath_scores.get(cat, 0.0) for cat in self.all_categories},
                "word_count": len(processed_text.split()),
                "avg_word_length": sum(len(word) for word in processed_text.split()) / len(processed_text.split()) if processed_text else 0,
                "sentence_count": len(re.split(r'[.!?]+', text)) if text else 0
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            return {}
    
    def get_category_stats(self, text: str) -> Dict[str, float]:
        """
        Get comprehensive statistics about text categories using Empath.
        
        Returns:
            Dictionary with various statistics including:
            - Category scores
            - Word count
            - Sentence count
            - Most prominent emotions
        """
        analysis = self.analyze_text(text)
        
        # Get most prominent emotions
        emotion_categories = [cat for cat in self.all_categories if "emotion" in cat]
        emotions = {cat: analysis.get(cat, 0) for cat in emotion_categories}
        
        stats = {
            "total_words": analysis.get("word_count", 0),
            "total_sentences": analysis.get("sentence_count", 0),
            "category_scores": analysis,
            "most_prominent_emotion": max(emotions.items(), key=lambda x: x[1])[0] if emotions else None,
            "avg_word_length": analysis.get("avg_word_length", 0)
        }
        
        return stats
