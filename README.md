# 📄 Smart Resume Screening System

An AI-powered resume screening tool built with Python that automatically analyzes and ranks
candidates based on their similarity to a given job description using **TF-IDF Vectorization**
and **Cosine Similarity**.

---

## 🚀 Features

- Upload multiple resumes in **PDF** or **TXT** format
- Enter a **Job Description** to match against
- Automatic **text extraction** and **NLP preprocessing**
- **TF-IDF vectorization** with bigrams for rich feature representation
- **Cosine Similarity** scoring for each candidate
- **Ranked results table** with color-coded match levels
- **Bar chart visualization** of candidate scores
- **CSV download** of screening results

---

## 🗂️ Project Structure

```
smart_resume_screening/
│
├── app.py               # Main Streamlit web application
├── resume_parser.py     # PDF & TXT text extraction module
├── similarity_model.py  # TF-IDF & cosine similarity engine
├── preprocessing.py     # Text cleaning & NLP preprocessing
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── sample_resumes/      # Sample resume files for testing
    ├── Alice_Johnson.txt
    ├── Bob_Martinez.txt
    └── Carol_Nguyen.txt
```

---

## ⚙️ Installation & Setup

### Step 1 — Prerequisites

Make sure you have **Python 3.9+** installed:
```bash
python --version
```

### Step 2 — Clone or Download the Project

```bash
cd C:\Users\Admin\.gemini\antigravity\scratch\smart_resume_screening
```

### Step 3 — (Recommended) Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `PyMuPDF` | PDF text extraction |
| `nltk` | Stopword removal & tokenization |
| `scikit-learn` | TF-IDF vectorizer & cosine similarity |
| `pandas` | Data handling |
| `numpy` | Numerical operations |
| `matplotlib` | Bar chart visualization |

### Step 5 — Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at **http://localhost:8501**

---

## 🧪 How to Test with Sample Resumes

1. Launch the app with `streamlit run app.py`
2. Paste the following sample **Job Description** in the text box:

```
We are looking for a Data Scientist with experience in Python, machine learning, NLP,
TF-IDF, scikit-learn, and data visualization. The candidate should have knowledge of
cosine similarity, text preprocessing, NLTK, and statistical modeling. Experience with
Pandas, NumPy, and deploying ML models is a plus.
```

3. Click **Browse files** and upload all 3 files from `sample_resumes/`
4. Click **🚀 Screen Resumes**
5. View the ranked results — **Alice Johnson** and **Carol Nguyen** should rank highest
   because their resumes closely match the data science job description.

---

## 🧠 NLP Pipeline Explained

```
Raw Text (Resume / Job Description)
        ↓
  [1] Lowercase conversion
        ↓
  [2] Remove URLs, emails, punctuation, digits
        ↓
  [3] Word tokenization (NLTK)
        ↓
  [4] Stopword removal (NLTK English corpus)
        ↓
  [5] TF-IDF Vectorization (unigrams + bigrams, max 5000 features)
        ↓
  [6] Cosine Similarity (JD vector vs each resume vector)
        ↓
  [7] Sort & Rank candidates by similarity score
```

---

## 📊 Score Interpretation

| Score Range | Match Level | Recommendation |
|-------------|-------------|----------------|
| 0.60 – 1.00 | 🟢 Strong   | Shortlist for interview |
| 0.30 – 0.59 | 🟡 Moderate | Consider with review |
| 0.00 – 0.29 | 🔴 Weak     | Likely not a fit |

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Streamlit** — Web UI
- **PyMuPDF (fitz)** — PDF parsing
- **NLTK** — Natural Language Processing
- **Scikit-learn** — TF-IDF & Cosine Similarity
- **Pandas & NumPy** — Data manipulation
- **Matplotlib** — Visualization

---
AUTHOR - JANANI J 
VIT UNIVERSITY, VELLORE
## 👨‍🎓 About

**Smart Resume Screening System** — Final Year Project  
Built to demonstrate practical applications of Natural Language Processing and
Information Retrieval techniques in Human Resources automation.

