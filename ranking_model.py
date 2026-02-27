"""
ranking_model.py
================
Core AI Ranking Module for the Smart Resume Screening System.

This module handles:
  1. TF-IDF Vectorization  — Converts text to numerical vectors
  2. Cosine Similarity     — Measures how similar each resume is to the JD
  3. Candidate Ranking     — Sorts candidates from best to worst match
  4. Duplicate Detection   — Flags resumes that are too similar to each other
  5. Recommendation System — Selects the top candidates automatically

Why TF-IDF + Cosine Similarity?
  - TF-IDF gives importance weight to rare but relevant words (e.g., "TensorFlow")
    and reduces weight for very common words (e.g., "the", "is").
  - Cosine similarity measures directional similarity between two document vectors,
    making it scale-invariant (works regardless of resume length).
"""

import numpy  as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

from preprocessing    import clean_text
from skill_extractor  import extract_skills, analyze_resume_quality, get_quality_label


def build_tfidf_matrix(documents: list) -> tuple:
    """
    Creates a TF-IDF matrix from a list of text documents.

    TF-IDF (Term Frequency–Inverse Document Frequency):
      - TF  = how often a word appears in this document
      - IDF = how rare the word is across all documents
      - TF-IDF = TF × IDF → high for unique, relevant words

    Parameters:
        documents (list[str]): List of cleaned text strings.

    Returns:
        tuple: (tfidf_matrix, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features = 8000,    # Limit vocabulary to top 8000 terms
        ngram_range  = (1, 2),  # Include unigrams ('python') and bigrams ('machine learning')
        sublinear_tf = True,    # Use log(TF) to reduce impact of very frequent words
        min_df       = 1        # Include terms that appear in at least 1 document
    )

    # Fit (learn vocabulary) and transform (encode) all documents
    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix, vectorizer


def compute_cosine_similarity(jd_vector, resume_vectors) -> np.ndarray:
    """
    Computes cosine similarity between the job description vector
    and all resume vectors.

    Cosine Similarity = cos(θ) between two vectors.
    Range: 0.0 (no similarity) to 1.0 (identical).

    Parameters:
        jd_vector      : TF-IDF vector for job description (shape: 1 × vocab)
        resume_vectors : TF-IDF matrix for resumes (shape: n × vocab)

    Returns:
        np.ndarray: 1D array of similarity scores, one per resume.
    """
    # cosine_similarity returns a 2D matrix; we take the first (only) row
    scores = cosine_similarity(jd_vector, resume_vectors)[0]
    return scores


def rank_candidates(job_description: str, resumes: list) -> pd.DataFrame:
    """
    Main ranking function — orchestrates the full analysis pipeline.

    Pipeline:
      1. Preprocess JD and all resumes
      2. Build TF-IDF matrix
      3. Compute cosine similarity scores
      4. Extract skills from each resume
      5. Analyze resume quality
      6. Sort by similarity score (descending)
      7. Assign ranks

    Parameters:
        job_description (str): Raw HR-entered job description text.
        resumes (list[dict]): List of {'name', 'text'} dicts from resume_parser.

    Returns:
        pd.DataFrame: Ranked results with all analysis columns.
    """

    # ── Step 1: Preprocess Job Description ───────────────────────────────────
    cleaned_jd = clean_text(job_description)

    # ── Step 2: Preprocess Resume Texts ──────────────────────────────────────
    candidate_names  = []
    cleaned_resumes  = []
    raw_resume_texts = []

    for resume in resumes:
        candidate_names.append(resume['name'])
        cleaned_resumes.append(clean_text(resume['text']))
        raw_resume_texts.append(resume['text'])  # Keep raw text for quality analysis

    # ── Step 3: Build TF-IDF Corpus ──────────────────────────────────────────
    # Position 0 = Job Description, positions 1..n = Resumes
    corpus       = [cleaned_jd] + cleaned_resumes
    tfidf_matrix, _ = build_tfidf_matrix(corpus)

    # ── Step 4: Extract Vectors from Matrix ──────────────────────────────────
    jd_vector      = tfidf_matrix[0:1]   # Shape: (1, vocab_size)
    resume_vectors = tfidf_matrix[1:]    # Shape: (n_resumes, vocab_size)

    # ── Step 5: Compute Cosine Similarity ────────────────────────────────────
    similarity_scores = compute_cosine_similarity(jd_vector, resume_vectors)

    # ── Step 6: Extract Skills from Each Resume ───────────────────────────────
    all_skills = [
        extract_skills(raw_text)
        for raw_text in raw_resume_texts
    ]

    # Format skills as comma-separated string for table display
    skills_str = [
        ', '.join(skills) if skills else 'None detected'
        for skills in all_skills
    ]

    # ── Step 7: Analyze Resume Quality ───────────────────────────────────────
    quality_results = [
        analyze_resume_quality(raw_text)
        for raw_text in raw_resume_texts
    ]

    quality_scores = [r['score']                    for r in quality_results]
    quality_labels = [get_quality_label(r['score']) for r in quality_results]
    quality_details = quality_results  # Keep full details for breakdown display

    # ── Step 8: Build Results DataFrame ──────────────────────────────────────
    results_df = pd.DataFrame({
        'Candidate Name'   : candidate_names,
        'Similarity Score' : np.round(similarity_scores, 4),
        'Extracted Skills' : skills_str,
        'Skill Count'      : [len(s) for s in all_skills],
        'Quality Score'    : quality_scores,
        'Quality Label'    : quality_labels,
        '_skills_list'     : all_skills,      # Hidden column for charts
        '_quality_detail'  : quality_details  # Hidden column for breakdown
    })

    # ── Step 9: Sort by Similarity Score Descending, then Quality Score ───────
    results_df = results_df.sort_values(
        by        = ['Similarity Score', 'Quality Score'],
        ascending = [False, False]
    ).reset_index(drop=True)

    # ── Step 10: Assign Rank (1 = best match) ────────────────────────────────
    results_df.insert(0, 'Rank', range(1, len(results_df) + 1))

    return results_df


def detect_duplicates(resumes: list, threshold: float = 0.90) -> list:
    """
    Detects pairs of resumes that are suspiciously similar to each other.

    This can catch scenarios where the same resume was uploaded twice
    with different names, or near-identical resumes are submitted.

    Parameters:
        resumes   (list[dict])  : List of parsed resume dicts {'name', 'text'}.
        threshold (float)       : Similarity threshold above which resumes
                                  are flagged as duplicates (default: 0.90).

    Returns:
        list[dict]: Duplicate pairs as {'candidate_a', 'candidate_b', 'similarity'}.
    """
    if len(resumes) < 2:
        return []  # Need at least 2 resumes to compare

    # Preprocess all resume texts
    cleaned_texts = [clean_text(r['text']) for r in resumes]
    names         = [r['name'] for r in resumes]

    # Build TF-IDF matrix for all resumes only (no JD)
    tfidf_matrix, _ = build_tfidf_matrix(cleaned_texts)

    # Compute pairwise cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)

    duplicates = []

    # Check every unique pair (i, j) where i < j to avoid double counting
    n = len(resumes)
    for i in range(n):
        for j in range(i + 1, n):
            sim_score = round(float(sim_matrix[i][j]), 4)

            if sim_score >= threshold:
                duplicates.append({
                    'candidate_a': names[i],
                    'candidate_b': names[j],
                    'similarity' : sim_score
                })

    return duplicates


def get_top_recommendations(results_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Returns the top N recommended candidates from the ranked results.

    These are the best-matching candidates that the system recommends
    for the next stage (e.g., phone screening or interview).

    Parameters:
        results_df (pd.DataFrame): Full ranked results from rank_candidates().
        top_n      (int)          : Number of top candidates to recommend.

    Returns:
        pd.DataFrame: Subset of the top N candidates.
    """
    # Results are already sorted by rank; just take the first N rows
    recommendations = results_df.head(top_n).copy()
    return recommendations
