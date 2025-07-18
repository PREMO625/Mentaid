from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from typing import List

class NLPModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        """Load pre-trained NLP model"""
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'MODELS MADE', 'NLP TRANSFORMER')
        
        try:
            # Force CPU device to avoid meta tensor issues
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_path,
                num_labels=2,
                torch_dtype=torch.float32,  # Use float32 for CPU
            
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_path)
            self.model.eval()  # Set model to evaluation mode
        except OSError as e:
            raise Exception(f"NLP model files not found at {base_path}. Error: {str(e)}")
    
    def chunk_text(self, text: str, max_input_length: int = 512, overlap_size: int = 50) -> List[str]:
        """
        Split text into overlapping chunks suitable for the transformer model.
        
        Args:
            text (str): Input text
            max_input_length (int): Maximum length of each chunk in tokens
            overlap_size (int): Number of overlapping tokens between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        chunks = []
        step_size = max_input_length - overlap_size
        
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i : i + max_input_length]
            decoded_chunk = self.tokenizer.decode(chunk_tokens)
            re_encoded_chunk = self.tokenizer.encode(decoded_chunk, 
                                                   add_special_tokens=True, 
                                                   max_length=max_input_length, 
                                                   truncation=True)
            chunks.append(self.tokenizer.decode(re_encoded_chunk))
        
        return chunks
    
    def predict(self, text: str) -> float:
        """
        Predict mental health score using NLP model.
        
        Args:
            text (str): Input text
            
        Returns:
            float: Prediction score (0 to 1)
        """
        # Tokenize input
        inputs = self.tokenizer(text,
                              return_tensors='pt',
                              max_length=512,
                              padding=True)
        
        with torch.no_grad():
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            outputs = self.model(**inputs)
            
        # Get probability of positive class
        probabilities = torch.softmax(outputs.logits, dim=1)
        return probabilities[0][1].item()
    
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
