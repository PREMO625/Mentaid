import numpy as np
from transformers import AutoTokenizer

def chunk_text_svm(text, chunk_size=931, overlap=0.5):
    """
    Chunk text for SVM model with specified chunk size and overlap.
    
    Args:
        text (str): Input text
        chunk_size (int): Size of each chunk in characters
        overlap (float): Overlap ratio between chunks (0 to 1)
        
    Returns:
        list: List of chunks
    """
    chunks = []
    text_length = len(text)
    step_size = int(chunk_size * (1 - overlap))
    
    for i in range(0, text_length, step_size):
        chunk_end = min(i + chunk_size, text_length)
        chunk = text[i:chunk_end]
        chunks.append(chunk)
    
    return chunks

def chunk_text_nlp(text, max_tokens=512, tokenizer_name='bert-base-uncased'):
    """
    Chunk text for NLP transformer model using token-based splitting.
    
    Args:
        text (str): Input text
        max_tokens (int): Maximum number of tokens per chunk
        tokenizer_name (str): Name of the transformer tokenizer
        
    Returns:
        list: List of chunks
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def average_predictions(predictions):
    """
    Calculate average prediction from multiple chunks.
    
    Args:
        predictions (list): List of predictions (0 to 1)
        
    Returns:
        float: Average prediction
    """
    return np.mean(predictions)
