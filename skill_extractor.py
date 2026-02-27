"""
skill_extractor.py
==================
Skill Extraction and Resume Quality Analysis Module.

This module does two things:

1. SKILL EXTRACTION
   Scans resume text for a predefined list of technical and soft skills.
   Returns a list of matched skills found in the resume.

2. RESUME QUALITY SCORING
   Evaluates resume completeness based on the presence of key sections:
   - Contact Information (email, phone)
   - Skills Section
   - Education
   - Work Experience
   - Summary / Objective
   - Certifications / Projects
   Returns a quality score out of 10.
"""

import re


# ── Master Skill Dictionary ───────────────────────────────────────────────────
# Organized by category for better skill distribution analysis.
# Each entry is a skill keyword to search for in resume text.

SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "r", "go",
        "rust", "scala", "kotlin", "swift", "php", "ruby", "matlab"
    ],
    "Web Development": [
        "html", "css", "react", "angular", "vue", "node", "django", "flask",
        "fastapi", "bootstrap", "tailwind", "jquery", "rest api", "graphql"
    ],
    "Data Science & ML": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "tensorflow", "keras", "pytorch", "scikit-learn", "sklearn", "xgboost",
        "pandas", "numpy", "scipy", "data science", "computer vision", "bert",
        "transformer", "neural network", "reinforcement learning"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle",
        "cassandra", "elasticsearch", "nosql", "firebase"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins",
        "terraform", "ansible", "linux", "git", "github", "gitlab", "devops"
    ],
    "Data Analytics & BI": [
        "tableau", "power bi", "excel", "looker", "data visualization",
        "matplotlib", "seaborn", "plotly", "statistics", "data analysis"
    ],
    "Soft Skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "project management", "agile", "scrum",
        "time management", "collaboration"
    ]
}

# Flattened list of all skills for quick lookup
ALL_SKILLS = [
    skill
    for skills in SKILL_CATEGORIES.values()
    for skill in skills
]


def extract_skills(text: str) -> list:
    """
    Identifies technical and soft skills mentioned in resume text.

    The function does a case-insensitive search for each known skill
    using word-boundary matching to avoid false positives
    (e.g., "R" matching "for" or "her").

    Parameters:
        text (str): Raw or cleaned resume text.

    Returns:
        list[str]: Alphabetically sorted list of found skills.
    """
    if not text:
        return []

    # Convert text to lowercase for case-insensitive comparison
    text_lower = text.lower()

    found_skills = []

    for skill in ALL_SKILLS:
        # Use regex word boundaries for accurate matching
        # re.escape handles special chars like "c++" or "ci/cd"
        pattern = r'\b' + re.escape(skill) + r'\b'

        if re.search(pattern, text_lower):
            # Store the skill in proper title case for display
            found_skills.append(skill.title())

    # Return unique, sorted list of found skills
    return sorted(set(found_skills))


def get_skills_by_category(found_skills: list) -> dict:
    """
    Groups the extracted skills by their category.

    Parameters:
        found_skills (list): List of found skill strings (title case).

    Returns:
        dict: {category_name: [list of skills in that category]}
    """
    categorized = {}

    for category, skills in SKILL_CATEGORIES.items():
        matched = [
            s.title() for s in skills
            if s.title() in found_skills
        ]
        if matched:
            categorized[category] = matched

    return categorized


# ── Resume Quality Scoring ────────────────────────────────────────────────────

# Each quality indicator contributes points toward the total score of 10.
QUALITY_CHECKS = {
    "Contact Information": {
        "weight"  : 2.0,
        "patterns": [
            r'[\w\.-]+@[\w\.-]+\.\w+',  # Email address
            r'(\+?\d[\d\s\-().]{7,}\d)',  # Phone number
        ]
    },
    "Skills Section": {
        "weight"  : 2.0,
        "patterns": [
            r'\bskills?\b',
            r'\btechnical skills\b',
            r'\bcore competencies\b',
            r'\bproficiencies\b'
        ]
    },
    "Education": {
        "weight"  : 2.0,
        "patterns": [
            r'\beducation\b',
            r'\bdegree\b',
            r'\bbachelor\b',
            r'\bmaster\b',
            r'\bphd\b',
            r'\buniversity\b',
            r'\bcollege\b',
        ]
    },
    "Work Experience": {
        "weight"  : 2.0,
        "patterns": [
            r'\bexperience\b',
            r'\bwork history\b',
            r'\bemployment\b',
            r'\bjob\b',
            r'\bintern\b',
            r'\bposition\b',
        ]
    },
    "Summary / Objective": {
        "weight"  : 1.0,
        "patterns": [
            r'\bsummary\b',
            r'\bobjective\b',
            r'\bprofile\b',
            r'\babout me\b',
        ]
    },
    "Projects / Certifications": {
        "weight"  : 1.0,
        "patterns": [
            r'\bproject\b',
            r'\bcertif',
            r'\bachievement\b',
            r'\baward\b',
        ]
    }
}


def analyze_resume_quality(text: str) -> dict:
    """
    Evaluates resume quality by checking for the presence of key sections.

    Parameters:
        text (str): Raw resume text (not preprocessed, to retain section headers).

    Returns:
        dict: {
            'score'    : float (0.0 – 10.0),
            'breakdown': {section_name: {'found': bool, 'weight': float}},
            'feedback' : list of missing section recommendations
        }
    """
    if not text:
        return {'score': 0.0, 'breakdown': {}, 'feedback': ["No text found in resume."]}

    text_lower = text.lower()

    total_weight = sum(v['weight'] for v in QUALITY_CHECKS.values())
    earned_score = 0.0
    breakdown    = {}
    feedback     = []

    for section, config in QUALITY_CHECKS.items():
        # Check if any pattern for this section is found in the resume text
        found = any(
            re.search(pattern, text_lower)
            for pattern in config['patterns']
        )

        if found:
            earned_score += config['weight']
        else:
            feedback.append(f"Add a '{section}' section to improve your resume score.")

        breakdown[section] = {
            'found' : found,
            'weight': config['weight']
        }

    # Normalize to a score out of 10
    final_score = round((earned_score / total_weight) * 10, 1)

    return {
        'score'    : final_score,
        'breakdown': breakdown,
        'feedback' : feedback
    }


def get_quality_label(score: float) -> str:
    """
    Returns a human-readable quality label based on the score.

    Parameters:
        score (float): Quality score out of 10.

    Returns:
        str: Label such as 'Excellent', 'Good', 'Average', or 'Needs Work'.
    """
    if score >= 8.0:
        return "🌟 Excellent"
    elif score >= 6.0:
        return "✅ Good"
    elif score >= 4.0:
        return "⚠️ Average"
    else:
        return "❌ Needs Work"
