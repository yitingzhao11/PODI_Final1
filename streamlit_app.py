import streamlit as st

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="Mortgage Dashboard",
    page_icon="🏠",
    layout="wide"
)

# ==========================================================
# Import Pages
# ==========================================================

from src import page_intro
from src import page_visualization
from src import page_prediction
from src import page_explainability
from src import page_tuning
from src import wandb_tracker

# ==========================================================
# Sidebar Navigation
# ==========================================================

st.sidebar.title("Navigation")

pages = {

    "🏦 Business Case & Data":
        page_intro,

    "📊 Data Visualization":
        page_visualization,

    "🤖 Model Prediction":
        page_prediction,

    "🔍 Explainability (SHAP)":
        page_explainability,

    "⚙️ Hyperparameter Tuning":
        page_tuning
}

with st.sidebar:

    st.markdown("## PODI Final Project")

    st.markdown("---")

    selected_page = st.radio(
        "Navigate",
        list(pages.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")

    try:
        wandb_tracker.status_badge()
    except:
        st.caption(
            "W&B Tracking: OFF"
        )

    st.caption(
        "Group Members:\nAina, Joey, Tony, Yiting"
    )

# ==========================================================
# Load Selected Page
# ==========================================================

pages[selected_page].render()