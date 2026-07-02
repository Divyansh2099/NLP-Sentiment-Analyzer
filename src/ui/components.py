"""
Reusable Streamlit UI components for the Sentiment Analyzer.
"""

import streamlit as st


# ── Color Constants (matching portfolio theme) ────────────
BG_COLOR = "#0a0a0a"
SURFACE_COLOR = "#161616"
ACCENT_COLOR = "#14f0c4"
ACCENT2_COLOR = "#f97316"
TEXT_COLOR = "#ffffff"
SUBTEXT_COLOR = "#a0a0a0"
LINE_COLOR = "#2a2a2a"

POSITIVE_COLOR = "#22c55e"
NEUTRAL_COLOR = "#eab308"
NEGATIVE_COLOR = "#ef4444"

# ── Sentiment Config ────────────────────────────────────
SENTIMENT_CONFIG = {
    "positive": {"emoji": "😊", "color": POSITIVE_COLOR, "label": "Positive"},
    "neutral": {"emoji": "😐", "color": NEUTRAL_COLOR, "label": "Neutral"},
    "negative": {"emoji": "😞", "color": NEGATIVE_COLOR, "label": "Negative"},
}


def apply_custom_css() -> None:
    """Inject custom CSS matching the portfolio dark theme."""
    st.markdown(f"""
    <style>
        /* ── Base ──────────────────────────────────────── */
        .stApp {{
            background-color: {BG_COLOR};
            color: {TEXT_COLOR};
        }}
        /* ── Headers ───────────────────────────────────── */
        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_COLOR} !important;
            font-family: 'Space Grotesk', sans-serif;
        }}
        /* ── Text ──────────────────────────────────────── */
        .stMarkdown, p, span, label {{
            color: {SUBTEXT_COLOR} !important;
        }}
        /* ── Cards ────────────────────────────────────── */
        .result-card {{
            background-color: {SURFACE_COLOR};
            border: 1px solid {LINE_COLOR};
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
        }}
        /* ── Badges ────────────────────────────────────── */
        .language-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            background-color: {ACCENT_COLOR}20;
            color: {ACCENT_COLOR};
            border: 1px solid {ACCENT_COLOR}40;
        }}
        .translated-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            background-color: {ACCENT2_COLOR}20;
            color: {ACCENT2_COLOR};
            border: 1px solid {ACCENT2_COLOR}40;
        }}
        /* ── Sentiment Label ────────────────────────────── */
        .sentiment-label {{
            font-size: 28px;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }}
        /* ── Metric Cards ──────────────────────────────── */
        .metric-card {{
            background-color: {SURFACE_COLOR};
            border: 1px solid {LINE_COLOR};
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }}
        .metric-label {{
            font-size: 13px;
            color: {SUBTEXT_COLOR};
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        /* ── Streamlit overrides ────────────────────────── */
        .stTextArea textarea, .stTextInput input {{
            background-color: {SURFACE_COLOR} !important;
            color: {TEXT_COLOR} !important;
            border-color: {LINE_COLOR} !important;
        }}
        .stButton > button {{
            background-color: {ACCENT_COLOR};
            color: {BG_COLOR};
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.2s;
        }}
        .stButton > button:hover {{
            background-color: #0fcfaf;
        }}
        .stSelectbox, .stTabs [data-baseweb="tab-list"] > div {{
            background-color: {SURFACE_COLOR};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {SURFACE_COLOR};
        }}
        /* ── Scrollbar ──────────────────────────────────── */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {BG_COLOR};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {LINE_COLOR};
            border-radius: 4px;
        }}
    </style>
    """, unsafe_allow_html=True)


def render_header() -> None:
    """Render the page header with title and badges."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🧠 NLP Sentiment Analyzer")
        st.markdown(
            "Real-time sentiment classification across **5 languages** "
            "powered by BERT."
        )
    with col2:
        st.markdown(
            '<div class="result-card" style="text-align:center;">'
            '<div class="metric-value" style="color:#14f0c4;">94%</div>'
            '<div class="metric-label">Accuracy</div></div>',
            unsafe_allow_html=True,
        )


def sentiment_card(result: dict) -> None:
    """Render a styled sentiment analysis result card.

    Args:
        result: Prediction result dictionary from SentimentPredictor.
    """
    sentiment = result.get("sentiment", "neutral")
    confidence = result.get("confidence", 0.0)
    scores = result.get("scores", {})
    language = result.get("language", "en")
    language_name = result.get("language_name", "English")
    was_translated = result.get("was_translated", False)
    processing_ms = result.get("processing_time_ms", 0.0)

    config = SENTIMENT_CONFIG.get(sentiment, SENTIMENT_CONFIG["neutral"])

    # Main result card
    st.markdown(
        f"""
        <div class="result-card">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;">
                <div>
                    <span style="font-size:48px;">{config['emoji']}</span>
                    <span class="sentiment-label" style="color:{config['color']};">
                        {config['label']}
                    </span>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:36px; font-weight:700; color:{config['color']};">
                        {confidence * 100:.1f}%
                    </div>
                    <div style="font-size:12px; color:{SUBTEXT_COLOR};">confidence</div>
                </div>
            </div>

            <div style="display:flex; gap:8px; margin-bottom:16px;">
                <span class="language-badge">🌐 {language_name} ({language})</span>
                {'<span class="translated-badge">🔄 Translated</span>' if was_translated else ''}
                <span style="color:{SUBTEXT_COLOR}; font-size:12px; margin-left:auto;">
                    ⚡ {processing_ms:.1f}ms
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def example_reviews() -> dict[str, list[dict[str, str]]]:
    """Return example reviews in all 5 supported languages.

    Returns:
        Dictionary mapping language code to list of example dicts.
    """
    return {
        "English": [
            {"text": "This product exceeded all my expectations! Absolutely wonderful quality.", "sentiment": "positive"},
            {"text": "Terrible customer service. Waited 2 hours and still no resolution.", "sentiment": "negative"},
            {"text": "The item arrived on time and was as described. Nothing special though.", "sentiment": "neutral"},
        ],
        "Spanish": [
            {"text": "¡Este restaurante es increíble! La mejor comida que he probado.", "sentiment": "positive"},
            {"text": "Pésima experiencia, el hotel estaba sucio y el personal grosero.", "sentiment": "negative"},
            {"text": "El producto es aceptable, cumple su función básica sin más.", "sentiment": "neutral"},
        ],
        "French": [
            {"text": "Absolument magnifique ! Je recommande vivement ce produit à tout le monde.", "sentiment": "positive"},
            {"text": "Très déçu de cet achat, la qualité est loin d'être au rendez-vous.", "sentiment": "negative"},
            {"text": "C'est correct, ni bon ni mauvais. Un produit standard.", "sentiment": "neutral"},
        ],
        "German": [
            {"text": "Hervorragende Qualität und schneller Versand! Bin sehr zufrieden.", "sentiment": "positive"},
            {"text": "Völlig unzufrieden. Das Produkt ist defekt und der Kundenservice nicht erreichbar.", "sentiment": "negative"},
            {"text": "Das Produkt ist in Ordnung, entspricht den Erwartungen.", "sentiment": "neutral"},
        ],
        "Portuguese": [
            {"text": "Excelente! Adorei o produto, superou todas as minhas expectativas!", "sentiment": "positive"},
            {"text": "Péssimo atendimento, demorou semanas e o produto veio errado.", "sentiment": "negative"},
            {"text": "O produto é razoável, funciona mas não tem nada de especial.", "sentiment": "neutral"},
        ],
    }


def example_selector() -> tuple[str, str]:
    """Render a language and example selector.

    Returns:
        (selected_language, selected_text)
    """
    examples = example_reviews()
    languages = list(examples.keys())

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_lang = st.selectbox("Language", languages, key="example_lang")
    with col2:
        options = [f"[{e['sentiment']}] {e['text'][:60]}..." for e in examples[selected_lang]]
        selected_idx = st.selectbox("Example", range(len(options)), format_func=lambda i: options[i])
        selected_text = examples[selected_lang][selected_idx]["text"]

    return selected_lang, selected_text
