"""
Streamlit Web Application for the NLP Sentiment Analyzer.
Provides an interactive dark-themed UI for real-time sentiment analysis.

Run with: streamlit run src/ui/app.py
"""

import io
import time

import pandas as pd
import streamlit as st

from src.model.predictor import SentimentPredictor, reset_predictor
from src.ui.components import (
    apply_custom_css,
    render_header,
    sentiment_card,
    example_reviews,
    example_selector,
)
from src.ui.visualizations import (
    plot_confidence_gauge,
    plot_confidence_bars,
    plot_batch_distribution,
    plot_confidence_histogram,
)
from src.utils.logger import setup_logger

logger = setup_logger("ui.app")

# ── Page Configuration ───────────────────────────────────
st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apply Theme ────────────────────────────────────────
apply_custom_css()

# ── Session State ──────────────────────────────────────
if "predictor" not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False


def load_predictor():
    """Load the predictor model (with status tracking)."""
    if st.session_state.predictor is None:
        with st.spinner("Loading sentiment model... (this may take a moment)"):
            start = time.time()
            st.session_state.predictor = SentimentPredictor()
            st.session_state.load_time = time.time() - start
            st.session_state.model_loaded = True
            st.success(f"✅ Model loaded in {st.session_state.load_time:.2f}s")


# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter results below this confidence level.",
    )

    show_scores = st.toggle("Show per-class scores", value=True)
    show_language = st.toggle("Show language info", value=True)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Model**: BERT (multilingual-cased)
    - **Accuracy**: 94%+
    - **Languages**: 5 (EN, ES, FR, DE, PT)
    - **Classes**: Positive, Neutral, Negative
    """)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[API Docs](http://localhost:8000/docs)")
    st.markdown("[GitHub Repo](#)")

# ── Header ─────────────────────────────────────────────
render_header()

# ── Load Model ──────────────────────────────────────────
load_predictor()
predictor = st.session_state.predictor

# ── Main Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📋 Batch Analysis", "ℹ️ About"])

# ══════════════════════════════════════════════════════════
# Tab 1: Single Text Analysis
# ══════════════════════════════════════════════════════════
with tab1:
    # Example selector
    st.markdown("#### Or try an example:")
    selected_lang, example_text = example_selector()

    # Text input
    user_text = st.text_area(
        "Enter text for sentiment analysis:",
        value="",
        height=120,
        placeholder="Type or paste any text in English, Spanish, French, German, or Portuguese...",
        key="single_input",
    )

    col_fill, col_analyze = st.columns([5, 1])
    with col_analyze:
        analyze_clicked = st.button("🧠 Analyze", type="primary", use_container_width=True)

    # Use example button
    use_example = st.button(f"Use Example ({selected_lang})", key="use_example")

    if use_example:
        st.session_state.single_input = example_text
        st.rerun()

    if analyze_clicked and user_text.strip():
        with st.spinner("Analyzing..."):
            result = predictor.predict(user_text)

        # Result card
        sentiment_card(result)

        # Visualizations row
        col_gauge, col_bars = st.columns([1, 2])

        with col_gauge:
            gauge_fig = plot_confidence_gauge(result["confidence"], result["sentiment"])
            st.plotly_chart(gauge_fig, use_container_width=True, key="gauge")

        if show_scores:
            with col_bars:
                bars_fig = plot_confidence_bars(result["scores"])
                st.plotly_chart(bars_fig, use_container_width=True, key="bars")

        # Language info
        if show_language:
            st.markdown("---")
            lang_col1, lang_col2, lang_col3 = st.columns(3)
            with lang_col1:
                st.metric("Detected Language", result["language_name"])
            with lang_col2:
                st.metric("Language Code", result["language"])
            with lang_col3:
                translated = "Yes" if result["was_translated"] else "No"
                st.metric("Translated", translated)

        # Processing time
        st.caption(f"⚡ Processing time: {result['processing_time_ms']:.1f}ms")

# ══════════════════════════════════════════════════════════
# Tab 2: Batch Analysis
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Upload a CSV file or paste multiple texts:")

    batch_option = st.radio("Input method", ["Paste texts", "Upload CSV"], horizontal=True)

    texts_to_analyze = []

    if batch_option == "Paste texts":
        batch_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=200,
            placeholder="This product is amazing!\nTerrible quality, very disappointed.\nIt's okay, nothing special.\n...",
            key="batch_input",
        )
        if batch_input.strip():
            texts_to_analyze = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
    else:
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="CSV must have a 'text' column.",
        )
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if "text" in df.columns:
                    texts_to_analyze = df["text"].astype(str).tolist()
                    st.success(f"Loaded {len(texts_to_analyze)} texts from CSV")
                else:
                    st.error("CSV must have a 'text' column.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if texts_to_analyze:
        st.info(f"**{len(texts_to_analyze)}** texts ready for analysis")
        analyze_batch = st.button("🧠 Analyze Batch", type="primary")

        if analyze_batch:
            if len(texts_to_analyze) > 100:
                st.warning("Maximum 100 texts per batch. Analyzing first 100...")
                texts_to_analyze = texts_to_analyze[:100]

            with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                batch_result = predictor.predict_batch(texts_to_analyze)

            st.success(
                f"✅ Analyzed **{batch_result['count']}** texts in "
                f"**{batch_result['total_time_ms']:.1f}ms**"
            )

            # Summary charts
            col_dist, col_hist = st.columns(2)

            with col_dist:
                st.markdown("**Sentiment Distribution**")
                dist_fig = plot_batch_distribution(batch_result["results"])
                st.plotly_chart(dist_fig, use_container_width=True, key="batch_dist")

            with col_hist:
                st.markdown("**Confidence Distribution**")
                hist_fig = plot_confidence_histogram(batch_result["results"])
                st.plotly_chart(hist_fig, use_container_width=True, key="batch_hist")

            # Results table
            st.markdown("---")
            st.markdown("#### Results")

            table_data = []
            for r in batch_result["results"]:
                table_data.append({
                    "Text": r["text"][:80] + ("..." if len(r["text"]) > 80 else ""),
                    "Sentiment": r["sentiment"].upper(),
                    "Confidence": f"{r['confidence']*100:.1f}%",
                    "Language": r["language_name"],
                    "Translated": "Yes" if r["was_translated"] else "No",
                })

            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Sentiment": st.column_config.TextColumn("Sentiment"),
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
            )

            # Download button
            csv_buffer = io.StringIO()
            results_df = pd.DataFrame(table_data)
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )

# ══════════════════════════════════════════════════════════
# Tab 3: About / Model Info
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    ## About the NLP Sentiment Analyzer

    This tool uses a **BERT transformer model** (`bert-base-multilingual-cased`)
    fine-tuned on a curated dataset of **180K+ customer reviews** to classify
    text sentiment in real-time.

    ### 🎯 Key Features
    - **Real-time analysis**: Sub-100ms inference per text
    - **5 languages**: English, Spanish, French, German, Portuguese
    - **3 classes**: Positive, Neutral, Negative
    - **94%+ accuracy**: Validated on a held-out test set
    - **Batch processing**: Analyze up to 100 texts at once

    ### 🏗 How It Works
    1. **Language Detection**: Automatically detects the input language
    2. **Translation**: Non-English text is translated to English
    3. **Classification**: BERT model classifies sentiment
    4. **Results**: Confidence scores and label returned

    ### 📊 Training Dataset
    | Source | Type | Samples |
    |--------|------|---------|
    | Amazon Reviews | Product reviews | 100K |
    | Twitter Sentiment | Social media posts | 50K |
    | Hotel Reviews | Customer feedback | 30K |

    ### 🛠 Built With
    - **Python** · **PyTorch** · **Transformers** (HuggingFace)
    - **FastAPI** · **Streamlit** · **Plotly**
    - **BERT** (bert-base-multilingual-cased)

    ---
    *Built by **[Divyansh](https://divyansh2099.github.io/Portfolio/)** — Data/AI Engineer*
    """)
