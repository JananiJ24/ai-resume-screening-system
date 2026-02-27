"""
visualization.py
================
Data Visualization Module for the AI-Powered Smart Resume Screening System.

Generates three interactive charts using Plotly:
  1. Candidate Ranking Bar Chart  — similarity scores side by side
  2. Skill Distribution Pie Chart — most common skills across all resumes
  3. Resume Quality Comparison    — quality scores compared across candidates

Plotly is used for interactive charts (hover tooltips, zoom, download).
"""

import plotly.graph_objects as go
import plotly.express       as px
import pandas               as pd
from   collections          import Counter


# ── Color Palette ─────────────────────────────────────────────────────────────
# Consistent color scheme across all charts
RANK_COLORS = {
    1: '#f59e0b',  # Gold   — 1st place
    2: '#94a3b8',  # Silver — 2nd place
    3: '#b45309',  # Bronze — 3rd place
}
DEFAULT_COLOR      = '#3b82f6'   # Blue for remaining candidates
STRONG_MATCH_COLOR = '#16a34a'   # Green  (score ≥ 0.6)
MODERATE_COLOR     = '#d97706'   # Amber  (score 0.3–0.59)
WEAK_COLOR         = '#dc2626'   # Red    (score < 0.3)


def _score_color(score: float) -> str:
    """Returns a bar color based on the similarity score value."""
    if score >= 0.60:
        return STRONG_MATCH_COLOR
    elif score >= 0.30:
        return MODERATE_COLOR
    else:
        return WEAK_COLOR


def plot_candidate_ranking(results_df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive horizontal bar chart showing each candidate's
    cosine similarity score, color-coded by match strength.

    Parameters:
        results_df (pd.DataFrame): Ranked results from ranking_model.rank_candidates().

    Returns:
        plotly.graph_objects.Figure: Interactive bar chart.
    """

    # Sort so the highest-ranked candidate appears at the top of the chart
    df = results_df.sort_values('Similarity Score', ascending=True).copy()

    # Assign color to each bar based on match strength
    colors = df['Similarity Score'].apply(_score_color).tolist()

    # Medal labels for top 3
    rank_labels = {1: '🥇', 2: '🥈', 3: '🥉'}
    y_labels = [
        f"{rank_labels.get(int(row['Rank']), '')} {row['Candidate Name']}"
        for _, row in df.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x           = df['Similarity Score'],
        y           = y_labels,
        orientation = 'h',                  # Horizontal bars
        marker      = dict(
            color       = colors,
            line        = dict(color='white', width=1.5)
        ),
        text        = [f"{s:.2%}" for s in df['Similarity Score']],  # Show % labels
        textposition= 'outside',
        hovertemplate = (
            "<b>%{y}</b><br>"
            "Similarity Score: %{x:.4f}<br>"
            "Match: %{text}<extra></extra>"
        )
    ))

    # Add vertical threshold lines for reference
    fig.add_vline(x=0.60, line_dash="dash", line_color=STRONG_MATCH_COLOR,
                  annotation_text="Strong (60%)", annotation_position="top right",
                  line_width=1.5, opacity=0.6)
    fig.add_vline(x=0.30, line_dash="dash", line_color=MODERATE_COLOR,
                  annotation_text="Moderate (30%)", annotation_position="top right",
                  line_width=1.5, opacity=0.6)

    fig.update_layout(
        title       = dict(
            text    = "📊 Candidate Ranking by Similarity Score",
            font    = dict(size=16, color='#1e293b'),
            x       = 0.5
        ),
        xaxis       = dict(
            title   = "Cosine Similarity Score",
            tickformat='.0%',
            range   = [0, min(df['Similarity Score'].max() + 0.15, 1.05)],
            gridcolor='#e5e7eb'
        ),
        yaxis       = dict(title="Candidate"),
        plot_bgcolor= '#f8fafc',
        paper_bgcolor='#f8fafc',
        height      = max(350, len(df) * 55 + 120),
        margin      = dict(l=20, r=80, t=60, b=40),
        showlegend  = False
    )

    return fig


def plot_skill_distribution(results_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Creates a bar chart showing the most frequently occurring skills
    across all analyzed resumes.

    Parameters:
        results_df (pd.DataFrame): Must contain '_skills_list' column (list of skills per candidate).
        top_n      (int)          : Number of top skills to display (default 15).

    Returns:
        plotly.graph_objects.Figure: Interactive skill frequency bar chart.
    """

    # Flatten all skill lists into one big list
    all_skills = []
    for skill_list in results_df['_skills_list']:
        all_skills.extend(skill_list)

    if not all_skills:
        # Return empty figure with message if no skills found
        fig = go.Figure()
        fig.add_annotation(
            text="No skills detected in uploaded resumes.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#94a3b8')
        )
        fig.update_layout(title="Skill Distribution", height=350)
        return fig

    # Count frequency of each skill
    skill_counts = Counter(all_skills)
    top_skills   = skill_counts.most_common(top_n)

    skill_names  = [s[0] for s in top_skills]
    skill_freqs  = [s[1] for s in top_skills]

    # Color bars by frequency (more frequent = darker blue)
    max_freq  = max(skill_freqs)
    bar_colors = [
        f'rgba(59, 130, 246, {0.4 + 0.6 * (f / max_freq):.2f})'
        for f in skill_freqs
    ]

    fig = go.Figure(go.Bar(
        x           = skill_freqs,
        y           = skill_names,
        orientation = 'h',
        marker      = dict(color=bar_colors, line=dict(color='white', width=1)),
        text        = skill_freqs,
        textposition= 'outside',
        hovertemplate = "<b>%{y}</b><br>Found in %{x} resume(s)<extra></extra>"
    ))

    fig.update_layout(
        title       = dict(
            text    = "🔧 Top Skills Across All Resumes",
            font    = dict(size=16, color='#1e293b'),
            x       = 0.5
        ),
        xaxis       = dict(title="Number of Candidates", gridcolor='#e5e7eb'),
        yaxis       = dict(title="Skill", autorange='reversed'),
        plot_bgcolor= '#f8fafc',
        paper_bgcolor='#f8fafc',
        height      = max(400, len(top_skills) * 30 + 120),
        margin      = dict(l=20, r=60, t=60, b=40),
        showlegend  = False
    )

    return fig


def plot_quality_comparison(results_df: pd.DataFrame) -> go.Figure:
    """
    Creates a grouped bar chart comparing similarity scores and quality scores
    side-by-side for each candidate.

    This helps HR identify candidates who have both a high match score
    AND a well-structured resume.

    Parameters:
        results_df (pd.DataFrame): Ranked results with 'Quality Score' column.

    Returns:
        plotly.graph_objects.Figure: Grouped bar chart.
    """

    # Sort by rank so best candidates appear first
    df = results_df.sort_values('Rank').copy()

    fig = go.Figure()

    # Bar 1: Similarity Score (normalized ×10 for same scale as quality score)
    fig.add_trace(go.Bar(
        name    = '🎯 Similarity Score (×10)',
        x       = df['Candidate Name'],
        y       = df['Similarity Score'] * 10,
        marker  = dict(color='#3b82f6', opacity=0.85),
        text    = [f"{s:.2%}" for s in df['Similarity Score']],
        textposition = 'outside',
        hovertemplate = (
            "<b>%{x}</b><br>"
            "Similarity: %{text}<extra></extra>"
        )
    ))

    # Bar 2: Resume Quality Score (already on 0-10 scale)
    quality_colors = [
        STRONG_MATCH_COLOR if s >= 7 else (MODERATE_COLOR if s >= 5 else WEAK_COLOR)
        for s in df['Quality Score']
    ]
    fig.add_trace(go.Bar(
        name    = '📄 Resume Quality (out of 10)',
        x       = df['Candidate Name'],
        y       = df['Quality Score'],
        marker  = dict(color=quality_colors, opacity=0.85),
        text    = [f"{s}/10" for s in df['Quality Score']],
        textposition = 'outside',
        hovertemplate = (
            "<b>%{x}</b><br>"
            "Quality: %{text}<extra></extra>"
        )
    ))

    fig.update_layout(
        title      = dict(
            text   = "📋 Similarity Score vs Resume Quality Comparison",
            font   = dict(size=16, color='#1e293b'),
            x      = 0.5
        ),
        barmode    = 'group',
        xaxis      = dict(title="Candidate"),
        yaxis      = dict(title="Score (out of 10)", range=[0, 12], gridcolor='#e5e7eb'),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#f8fafc',
        height     = 420,
        margin     = dict(l=20, r=20, t=70, b=60),
        legend     = dict(
            orientation='h',
            yanchor='bottom', y=1.01,
            xanchor='right',  x=1
        )
    )

    return fig
