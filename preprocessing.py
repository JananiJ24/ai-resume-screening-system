"""
preprocessing.py
================
Text Preprocessing and Cleaning Module.

This module provides text normalization and cleaning utilities:
  - Lowercasing and whitespace normalization
  - Stopword removal
  - Tokenization
  - Special character handling

Clean text is prepared for TF-IDF vectorization and other NLP tasks.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Download required NLTK data (if not already present)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def clean_text(text: str) -> str:
    """
    Preprocesses text for TF-IDF vectorization and similarity matching.
    
    Steps:
      1. Lowercase all text
      2. Remove URLs
      3. Remove email addresses
      4. Remove special characters (keep alphanumeric and spaces)
      5. Remove extra whitespace
      6. Remove stopwords
      7. Tokenize and rejoin for consistency
    
    Parameters:
        text (str): Raw text to clean.
    
    Returns:
        str: Cleaned text ready for vectorization.
    """
    if not text:
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Step 3: Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Step 4: Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    # Keep: letters, numbers, spaces, and common separators like hyphen
    text = re.sub(r'[^a-z0-9\s\+\-\#]', ' ', text)
    
    # Step 5: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 6: Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        filtered_tokens = [
            token for token in tokens 
            if token not in stop_words and len(token) > 1
        ]
        text = ' '.join(filtered_tokens)
    except Exception:
        # Fallback if NLTK fails
        # Just ensure text is clean
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text
