"""
Visualization components for the Streamlit UI.
Uses Plotly for interactive charts.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Color Constants ────────────────────────────────────
POSITIVE_COLOR = "#22c55e"
NEUTRAL_COLOR = "#eab308"
NEGATIVE_COLOR = "#ef4444"
ACCENT_COLOR = "#14f0c4"
BG_COLOR = "#0a0a0a"
SURFACE_COLOR = "#161616"
LINE_COLOR = "#2a2a2a"
TEXT_COLOR = "#ffffff"
SUBTEXT_COLOR = "#a0a0a0"

COLORS = {
    "positive": POSITIVE_COLOR,
    "neutral": NEUTRAL_COLOR,
    "negative": NEGATIVE_COLOR,
}


def plot_confidence_gauge(confidence: float, sentiment: str) -> go.Figure:
    """Create a circular gauge chart showing confidence score.

    Args:
        confidence: Confidence value (0-1).
        sentiment: Predicted sentiment label.

    Returns:
        Plotly Figure object.
    """
    color = COLORS.get(sentiment, ACCENT_COLOR)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": "%",
                "font": {"size": 48, "color": color, "family": "Space Grotesk"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": SUBTEXT_COLOR,
                    "tickfont": {"size": 10, "color": SUBTEXT_COLOR},
                },
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": SURFACE_COLOR,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 100], "color": LINE_COLOR},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": confidence * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(t=20, b=20, l=20, r=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
    )

    return fig


def plot_confidence_bars(scores: dict[str, float]) -> go.Figure:
    """Create a horizontal bar chart of per-class confidence scores.

    Args:
        scores: Dictionary mapping label names to probabilities.

    Returns:
        Plotly Figure object.
    """
    labels = list(scores.keys())
    values = [scores[l] * 100 for l in labels]
    colors = [COLORS.get(l, ACCENT_COLOR) for l in labels]

    fig = go.Figure(
        go.Bar(
            orientation="h",
            y=labels,
            x=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=180,
        margin=dict(t=10, b=10, l=80, r=50),
        xaxis=dict(
            range=[0, 105],
            gridcolor=LINE_COLOR,
            tickfont=dict(color=SUBTEXT_COLOR, size=11),
        ),
        yaxis=dict(
            tickfont=dict(color=TEXT_COLOR, size=13, family="Space Grotesk"),
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        showlegend=False,
    )

    return fig


def plot_batch_distribution(results: list[dict]) -> go.Figure:
    """Create a pie chart of sentiment distribution across batch results.

    Args:
        results: List of prediction result dictionaries.

    Returns:
        Plotly Figure object.
    """
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        sentiment = r.get("sentiment", "neutral")
        counts[sentiment] = counts.get(sentiment, 0) + 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [COLORS.get(l, ACCENT_COLOR) for l in labels]

    fig = go.Figure(
        go.Pie(
            labels=[l.capitalize() for l in labels],
            values=values,
            marker=dict(colors=colors),
            hole=0.6,
            textfont=dict(color=TEXT_COLOR, size=13, family="Space Grotesk"),
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        showlegend=True,
        legend=dict(
            font=dict(color=SUBTEXT_COLOR, size=12),
            bgcolor=SURFACE_COLOR,
            bordercolor=LINE_COLOR,
        ),
    )

    return fig


def plot_confidence_histogram(results: list[dict]) -> go.Figure:
    """Create a histogram of confidence scores across batch results.

    Args:
        results: List of prediction result dictionaries.

    Returns:
        Plotly Figure object.
    """
    confidences = [r.get("confidence", 0) * 100 for r in results]

    fig = go.Figure(
        go.Histogram(
            x=confidences,
            nbinsx=20,
            marker_color=ACCENT_COLOR,
            opacity=0.8,
            hovertemplate="Range: %{x}%<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(t=10, b=40, l=40, r=20),
        xaxis=dict(
            title="Confidence (%)",
            titlefont=dict(color=SUBTEXT_COLOR),
            tickfont=dict(color=SUBTEXT_COLOR),
            gridcolor=LINE_COLOR,
        ),
        yaxis=dict(
            title="Count",
            titlefont=dict(color=SUBTEXT_COLOR),
            tickfont=dict(color=SUBTEXT_COLOR),
            gridcolor=LINE_COLOR,
        ),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        bargap=0.05,
    )

    return fig
