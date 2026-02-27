"""
app.py
======
Main Streamlit Application — AI-Powered Smart Resume Screening System

This is the entry point for the web application. It ties together all modules:
  - resume_parser    → Extracts text from uploaded PDF/TXT resumes
  - ranking_model    → Ranks candidates using TF-IDF + Cosine Similarity
  - skill_extractor  → Identifies skills and evaluates resume quality
  - visualization    → Generates interactive Plotly charts

Run this app with:
    streamlit run app.py
"""

import streamlit as st
import pandas    as pd

# ── Import project modules ─────────────────────────────────────────────────────
from resume_parser  import parse_all_resumes
from ranking_model  import rank_candidates, detect_duplicates, get_top_recommendations
from visualization  import (
    plot_candidate_ranking,
    plot_skill_distribution,
    plot_quality_comparison
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# Must be called as the first Streamlit command in the script.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Resume Screener",
    page_icon  = "🤖",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme Dashboard
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.8rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 1.8rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(99,102,241,0.15) 0%, transparent 60%);
}
.hero-header h1 {
    color: #f1f5f9;
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.8px;
}
.hero-header p {
    color: #a5b4fc;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #e2e8f0;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f0f4ff 0%, #e8ecfb 100%);
    border: 1px solid #dde3f5;
    border-radius: 12px;
    padding: 0.85rem 1rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* ── Top rec cards ── */
.rec-card {
    background: linear-gradient(135deg, #fefce8, #fef9c3);
    border: 1.5px solid #fbbf24;
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 12px rgba(251,191,36,0.15);
}
.rec-card h4 { margin: 0 0 0.4rem 0; color: #1e293b; font-size: 1rem; }
.rec-card p  { margin: 0; color: #475569; font-size: 0.88rem; }

/* ── Duplicate alert ── */
.dup-card {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
}

/* ── Skill badge ── */
.skill-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px 2px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Quality section row ── */
.quality-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.3rem 0; }
.q-found     { color: #16a34a; font-weight: 600; }
.q-missing   { color: #dc2626; font-weight: 600; }

/* ── Analyze button ── */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    box-shadow: 0 5px 18px rgba(79,70,229,0.4);
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(79,70,229,0.55);
}

/* ── Table styling ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🤖 AI-Powered Smart Resume Screening System</h1>
    <p>Automatically analyze, rank, and compare candidates using NLP &amp;
    TF-IDF cosine similarity &mdash; built for modern HR teams</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR PANEL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation Guide")
    st.info(
        "**1.** Enter a Job Description\n\n"
        "**2.** Upload resumes (PDF or TXT)\n\n"
        "**3.** Click **Analyze Resumes**\n\n"
        "**4.** Explore results, charts & recommendations"
    )

    st.markdown("---")
    st.markdown("## ⚙️ NLP Pipeline")
    steps = [
        ("📥", "Text Extraction",       "PDF / TXT parsing"),
        ("🔡", "Preprocessing",         "Lowercase, stopwords, punctuation"),
        ("🔧", "Skill Extraction",       "Keyword & pattern matching"),
        ("📐", "TF-IDF Vectorization",   "Bag-of-words with IDF weighting"),
        ("📏", "Cosine Similarity",      "Match score per candidate"),
        ("🏆", "Ranking",               "Sort by score, assign rank"),
        ("📋", "Quality Analysis",       "Section completeness score"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"**{icon} {title}**  \n<small>{desc}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🎯 Score Legend")
    st.markdown("""
| Score | Match Level |
|-------|-------------|
| ≥ 60% | 🟢 Strong   |
| 30–59%| 🟡 Moderate |
| < 30% | 🔴 Weak     |
    """)

    st.markdown("---")
    st.caption("📚 Final Year Project · AI Resume Screening System")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
col_jd, col_up = st.columns([1.15, 0.85], gap="large")

with col_jd:
    st.markdown('<p class="section-title">📝 Job Description</p>', unsafe_allow_html=True)
    job_description = st.text_area(
        label            = "Job Description",
        height           = 290,
        placeholder      = (
            "Paste the full job description here...\n\n"
            "Example:\nWe are looking for a Senior Data Scientist with 4+ years of "
            "experience in Python, machine learning, NLP, TensorFlow, scikit-learn, "
            "SQL, and data visualization. Strong communication skills and a background "
            "in statistics or computer science are essential."
        ),
        label_visibility = "collapsed"
    )

with col_up:
    st.markdown('<p class="section-title">📂 Upload Resumes</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label              = "Upload Resumes",
        type               = ["pdf", "txt"],
        accept_multiple_files = True,
        label_visibility   = "collapsed",
        help               = "Upload one or more PDF or TXT resume files"
    )

    # Show uploaded file list with icons
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) ready:**")
        for f in uploaded_files:
            icon = "📕" if f.name.lower().endswith(".pdf") else "📄"
            size = round(f.size / 1024, 1)
            st.markdown(f"{icon} `{f.name}` — {size} KB")

    # Duplicate detection threshold slider
    st.markdown("---")
    dup_threshold = st.slider(
        label  = "🔍 Duplicate Detection Threshold",
        min_value = 0.70, max_value = 1.00, value = 0.90, step = 0.01,
        help   = "Resumes above this similarity to each other are flagged as duplicates"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANALYZE BUTTON — centered below inputs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("")
btn_col = st.columns([1, 1.2, 1])[1]
with btn_col:
    analyze_btn = st.button("🔍 Analyze Resumes", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE — triggered when button is clicked
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn:

    # ── Input Validation ──────────────────────────────────────────────────────
    if not job_description.strip():
        st.warning("⚠️ Please enter a Job Description before analyzing.")
        st.stop()

    if not uploaded_files:
        st.warning("⚠️ Please upload at least one resume file.")
        st.stop()

    # ── Step 1: Parse Resumes ─────────────────────────────────────────────────
    with st.spinner("📥 Extracting text from resumes..."):
        resumes = parse_all_resumes(uploaded_files)

    if not resumes:
        st.error("❌ Could not extract text from any uploaded resume. Please check the files.")
        st.stop()

    # ── Step 2: Run AI Ranking Pipeline ──────────────────────────────────────
    with st.spinner("🧠 Running AI analysis — TF-IDF vectorization & ranking..."):
        results_df = rank_candidates(job_description, resumes)

    # ── Step 3: Detect Duplicates ─────────────────────────────────────────────
    with st.spinner("🔍 Checking for duplicate resumes..."):
        duplicates = detect_duplicates(resumes, threshold=dup_threshold)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY METRICS ROW
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 📊 Analysis Summary")
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("👥 Candidates", len(results_df))
    with m2:
        top_score = results_df['Similarity Score'].max()
        st.metric("🏆 Top Match", f"{top_score:.2%}")
    with m3:
        avg_score = results_df['Similarity Score'].mean()
        st.metric("📊 Avg Score", f"{avg_score:.2%}")
    with m4:
        strong = (results_df['Similarity Score'] >= 0.30).sum()
        st.metric("✅ Good Matches", strong)
    with m5:
        avg_quality = results_df['Quality Score'].mean()
        st.metric("📄 Avg Quality", f"{avg_quality:.1f}/10")

    st.markdown("")

    # ─────────────────────────────────────────────────────────────────────────
    # DUPLICATE DETECTION ALERT
    # ─────────────────────────────────────────────────────────────────────────
    if duplicates:
        st.markdown(f"### ⚠️ Duplicate Resumes Detected ({len(duplicates)} pair(s))")
        for dup in duplicates:
            st.markdown(f"""
<div class="dup-card">
    🔁 <strong>{dup['candidate_a']}</strong> and <strong>{dup['candidate_b']}</strong>
    are <strong>{dup['similarity']:.2%}</strong> similar — possible duplicate!
</div>
""", unsafe_allow_html=True)
    else:
        st.success(f"✅ No duplicate resumes detected (threshold: {dup_threshold:.0%}).")

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # TOP 3 RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("## 🌟 Top Candidate Recommendations")
    recommendations = get_top_recommendations(results_df, top_n=min(3, len(results_df)))

    medals = ["🥇", "🥈", "🥉"]
    rec_cols = st.columns(len(recommendations))

    for idx, (col, (_, row)) in enumerate(zip(rec_cols, recommendations.iterrows())):
        with col:
            medal = medals[idx] if idx < 3 else "🏅"
            score_pct = f"{row['Similarity Score']:.2%}"
            quality   = f"{row['Quality Score']}/10"
            top_skills = ', '.join(row['_skills_list'][:5]) if row['_skills_list'] else "—"

            st.markdown(f"""
<div class="rec-card">
    <h4>{medal} Rank #{int(row['Rank'])} — {row['Candidate Name']}</h4>
    <p>🎯 <b>Match:</b> {score_pct} &nbsp;|&nbsp; 📄 <b>Quality:</b> {quality}</p>
    <p>🔧 <b>Top Skills:</b> {top_skills}</p>
    <p>{row['Quality Label']}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────────
    # TABBED RESULTS SECTION
    # ─────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Ranked Table",
        "📊 Ranking Chart",
        "🔧 Skill Distribution",
        "📈 Quality Comparison",
        "🔎 Resume Details"
    ])

    # ── Tab 1: Ranked Table ───────────────────────────────────────────────────
    with tab1:
        st.markdown("### Candidate Ranking Table")

        # Build display DataFrame (hide internal columns)
        display_df = results_df[[
            'Rank', 'Candidate Name', 'Similarity Score',
            'Extracted Skills', 'Skill Count', 'Quality Score', 'Quality Label'
        ]].copy()

        # Add match level column
        display_df['Match Level'] = display_df['Similarity Score'].apply(
            lambda s: '🟢 Strong' if s >= 0.60 else ('🟡 Moderate' if s >= 0.30 else '🔴 Weak')
        )
        display_df['Similarity Score (%)'] = (display_df['Similarity Score'] * 100).round(2)

        # Style the DataFrame
        def color_score(val):
            if val >= 0.60:
                return 'background-color: #dcfce7; color: #15803d; font-weight:600'
            elif val >= 0.30:
                return 'background-color: #fef9c3; color: #92400e; font-weight:600'
            return 'background-color: #fee2e2; color: #b91c1c; font-weight:600'

        def color_rank(val):
            if val == 1: return 'background-color: #fef3c7; color: #b45309; font-weight:700'
            if val == 2: return 'background-color: #f1f5f9; color: #475569; font-weight:700'
            if val == 3: return 'background-color: #fdf4e7; color: #92400e; font-weight:700'
            return ''

        styled = (
            display_df[[
                'Rank', 'Candidate Name', 'Similarity Score (%)',
                'Match Level', 'Skill Count', 'Quality Score', 'Quality Label'
            ]]
            .style
            .applymap(color_score,  subset=['Similarity Score (%)'])  # type: ignore
            .applymap(color_rank,   subset=['Rank'])                   # type: ignore
            .format({'Similarity Score (%)': '{:.2f}%', 'Quality Score': '{:.1f}'})
            .set_properties(**{'text-align': 'center'})
            .set_table_styles([{
                'selector': 'th',
                'props': [
                    ('background-color', '#312e81'),
                    ('color', 'white'),
                    ('font-weight', '600'),
                    ('text-align', 'center'),
                    ('padding', '10px 14px')
                ]
            }])
        )
        st.dataframe(styled, use_container_width=True, height=420)

        # Download CSV button
        csv = results_df.drop(columns=['_skills_list', '_quality_detail']).to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Results as CSV",
            data      = csv,
            file_name = "resume_screening_results.csv",
            mime      = "text/csv"
        )

    # ── Tab 2: Ranking Chart ──────────────────────────────────────────────────
    with tab2:
        st.markdown("### Candidate Ranking — Similarity Scores")
        fig_rank = plot_candidate_ranking(results_df)
        st.plotly_chart(fig_rank, use_container_width=True)

    # ── Tab 3: Skill Distribution ─────────────────────────────────────────────
    with tab3:
        st.markdown("### Most Demanded Skills Across Resumes")
        top_n_skills = st.slider("Show top N skills:", 5, 25, 15, key="top_n")
        fig_skills = plot_skill_distribution(results_df, top_n=top_n_skills)
        st.plotly_chart(fig_skills, use_container_width=True)

    # ── Tab 4: Quality Comparison ─────────────────────────────────────────────
    with tab4:
        st.markdown("### Resume Quality vs Similarity Score")
        fig_quality = plot_quality_comparison(results_df)
        st.plotly_chart(fig_quality, use_container_width=True)

    # ── Tab 5: Resume Details ─────────────────────────────────────────────────
    with tab5:
        st.markdown("### Individual Resume Breakdown")

        selected_candidate = st.selectbox(
            "Select a candidate to view details:",
            options = results_df['Candidate Name'].tolist()
        )

        # Get selected candidate's row
        cand_row = results_df[results_df['Candidate Name'] == selected_candidate].iloc[0]
        quality_detail = cand_row['_quality_detail']

        d1, d2 = st.columns(2)

        with d1:
            st.markdown(f"#### 🎯 Similarity Score")
            score = cand_row['Similarity Score']
            match = '🟢 Strong Match' if score >= 0.60 else ('🟡 Moderate Match' if score >= 0.30 else '🔴 Weak Match')
            st.metric("Score", f"{score:.4f}")
            st.caption(f"{score:.2%} — {match}")

            st.markdown("#### 🔧 Extracted Skills")
            if cand_row['_skills_list']:
                skill_html = ' '.join(
                    f'<span class="skill-badge">{s}</span>'
                    for s in cand_row['_skills_list']
                )
                st.markdown(skill_html, unsafe_allow_html=True)
            else:
                st.info("No recognizable skills found.")

        with d2:
            st.markdown(f"#### 📄 Resume Quality: {cand_row['Quality Score']}/10 — {cand_row['Quality Label']}")
            st.progress(cand_row['Quality Score'] / 10)

            st.markdown("**Section Checklist:**")
            for section, info in quality_detail['breakdown'].items():
                icon = "✅" if info['found'] else "❌"
                st.markdown(f"{icon} {section} *(weight: {info['weight']})*")

            if quality_detail['feedback']:
                st.markdown("**💡 Improvement Tips:**")
                for tip in quality_detail['feedback']:
                    st.caption(f"• {tip}")


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE (when button not yet clicked)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("---")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem; background:#f8fafc; color:#1e293b; border-radius:14px; border:1px solid #e2e8f0;">
            <div style="font-size:2.5rem">📝</div>
            <b>Step 1</b><br>Enter the Job Description
        </div>""", unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem; background:#f8fafc; border-radius:14px; border:1px solid #e2e8f0;">
            <div style="font-size:2.5rem">📂</div>
            <b>Step 2</b><br>Upload Resume Files (PDF/TXT)
        </div>""", unsafe_allow_html=True)
    with colC:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem; background:#f8fafc; border-radius:14px; border:1px solid #e2e8f0;">
            <div style="font-size:2.5rem">🚀</div>
            <b>Step 3</b><br>Click Analyze Resumes
        </div>""", unsafe_allow_html=True)

