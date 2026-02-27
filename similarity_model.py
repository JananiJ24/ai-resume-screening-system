"""
similarity_model.py
-------------------
Core NLP model for the Smart Resume Screening System.

Process:
1. Combine the job description with all resume texts into a single corpus.
2. Apply TF-IDF Vectorization to convert text into numerical feature vectors.
3. Compute Cosine Similarity between the job description vector and each resume vector.
4. Rank candidates based on their similarity scores (highest = best match).

Why TF-IDF?
    TF-IDF (Term Frequency–Inverse Document Frequency) is a statistical measure
    that reflects how important a word is to a document relative to the corpus.
    It reduces the weight of common words and highlights domain-specific keywords.

Why Cosine Similarity?
    Cosine similarity measures the angle between two vectors in high-dimensional space.
    A score of 1.0 = identical, 0.0 = completely dissimilar. It is scale-invariant,
    making it ideal for comparing documents of different lengths.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text


def compute_similarity(job_description: str, resumes: list[dict]) -> pd.DataFrame:
    """
    Computes TF-IDF cosine similarity between a job description and multiple resumes.
    Returns a ranked DataFrame of candidates.

    Parameters:
        job_description (str): Raw job description text entered by the HR user.
        resumes (list[dict]): List of dicts with keys 'name' (str) and 'text' (str).

    Returns:
        pd.DataFrame: Ranked candidates with columns:
            - Rank            : Position (1 = best match)
            - Candidate Name  : Derived from resume filename
            - Similarity Score: Float between 0.0 and 1.0 (rounded to 4 decimals)
    """

    # ── Step 1: Preprocess the Job Description ────────────────────────────────
    cleaned_jd = clean_text(job_description)

    # ── Step 2: Preprocess Each Resume Text ──────────────────────────────────
    candidate_names = []
    cleaned_resume_texts = []

    for resume in resumes:
        candidate_names.append(resume['name'])
        cleaned_resume_texts.append(clean_text(resume['text']))

    # ── Step 3: Build the Corpus ──────────────────────────────────────────────
    # The job description is the FIRST document in the corpus.
    # All resume texts follow it.
    corpus = [cleaned_jd] + cleaned_resume_texts

    # ── Step 4: Apply TF-IDF Vectorization ───────────────────────────────────
    # TfidfVectorizer converts text documents into a matrix of TF-IDF features.
    # max_features=5000 limits vocabulary size for performance.
    vectorizer = TfidfVectorizer(
        max_features=5000,      # Limit vocabulary to top 5000 terms
        ngram_range=(1, 2),     # Include both unigrams and bigrams
        sublinear_tf=True       # Apply log normalization to TF values
    )

    # Fit the vectorizer on the corpus and transform all documents
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # ── Step 5: Extract Vectors ───────────────────────────────────────────────
    # Index 0  → Job Description vector
    # Index 1+ → Resume vectors
    jd_vector = tfidf_matrix[0:1]           # Shape: (1, vocab_size)
    resume_vectors = tfidf_matrix[1:]       # Shape: (n_resumes, vocab_size)

    # ── Step 6: Compute Cosine Similarity ────────────────────────────────────
    # Similarity of each resume against the job description
    similarity_scores = cosine_similarity(jd_vector, resume_vectors)[0]
    # Result: 1D array of shape (n_resumes,) with scores in [0.0, 1.0]

    # ── Step 7: Build Results DataFrame ──────────────────────────────────────
    results_df = pd.DataFrame({
        'Candidate Name': candidate_names,
        'Similarity Score': np.round(similarity_scores, 4)
    })

    # ── Step 8: Sort and Rank by Similarity Score (Descending) ───────────────
    results_df = results_df.sort_values(
        by='Similarity Score', ascending=False
    ).reset_index(drop=True)

    # Add rank column (1-indexed, higher score = lower rank number = better)
    results_df.insert(0, 'Rank', range(1, len(results_df) + 1))

    return results_df
