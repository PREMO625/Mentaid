import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sklearn
import joblib
from typing import List
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import load

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SVMModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_model()
        
    def load_model(self):
        """Load pre-trained SVM model and vectorizer"""
        models_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(models_dir, 'ACCESS_MODELS', 'SVM', 'best_svm_pipeline.pkl')
        
        try:
            # Check package versions
            print(f"Loading model with:")
            print(f"NumPy version: {np.__version__}")
            print(f"scikit-learn version: {sklearn.__version__}")
            print(f"joblib version: {joblib.__version__}")
            
            # Provide backward-compat alias for pickles created with NumPy>=2
            import sys
            import numpy.core as _ncore
            sys.modules['numpy._core'] = _ncore
            sys.modules['numpy._core.multiarray'] = _ncore.multiarray
            # Load the model using joblib
            pipeline = joblib.load(model_path)
            self.model = pipeline.named_steps['pipeline'].named_steps['svm']
            self.vectorizer = pipeline.named_steps['tfidfvectorizer']
            print(f"Successfully loaded model")
            return
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please ensure you have installed the exact versions of packages:")
            print("numpy==2.0.2")
            print("scikit-learn==1.6.1")
            print("joblib==1.5.1")
            raise Exception(f"Failed to load model: {str(e)}")
        
        except FileNotFoundError:
            raise Exception(f"SVM model file not found at {model_path}")
    
    def preprocess_text(self, text):
        """Preprocess text for SVM model"""
        if isinstance(text, str):
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Convert to lowercase
            text = text.lower()
            # Tokenize and remove stopwords and stem
            text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
            return text
        else:
            return ""  # Return empty string for non-string types
    
    def predict(self, text: str) -> float:
        """
        Predict mental health score using SVM model.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Prediction score (0 to 1)
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize text
        vectorized_text = self.vectorizer.transform([processed_text])
        
                # Predict probability or fallback to decision_function
                # Use predict_proba if model was trained with probability=True
        if getattr(self.model, "probability", False):
            prediction = self.model.predict_proba(vectorized_text)[0][1]
        else:
            # Use decision_function score and convert to pseudo-probability via sigmoid
            import math
            score = self.model.decision_function(vectorized_text)[0]
            prediction = 1 / (1 + math.exp(-score))
        return prediction
    
    def predict_chunks(self, chunks: List[str]) -> List[float]:
        """
        Predict scores for multiple chunks of text.
        
        Args:
            chunks (List[str]): List of text chunks
            
        Returns:
            List[float]: List of prediction scores
        """
        predictions = []
        for chunk in chunks:
            predictions.append(self.predict(chunk))
        return predictions
