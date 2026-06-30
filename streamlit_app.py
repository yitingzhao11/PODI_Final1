import streamlit as st

# ==========================================================
# Page Config (Must be the first Streamlit command)
# ==========================================================

st.set_page_config(
    page_title="Mortgage Dashboard",
    page_icon="🏠",
    layout="wide"
)

# ==========================================================
# Load Global CSS
# ==========================================================

def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

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
# Sidebar
# ==========================================================

with st.sidebar:

    st.image("assets/house.jpg", width = 300)

    st.markdown(
        "<h2 style='text-align:center;'>PODI Final Project</h2>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Interactive Prediction Models & Machine Learning for Mortgage Loan Prediction"
    )

    st.markdown("---")

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

    st.subheader("📂 Navigation")

    selected_page = st.radio(
        "",
        list(pages.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.subheader("📈 Experiment Tracking")

    try:
        wandb_tracker.status_badge()
    except:
        st.warning("W&B Offline")

    st.markdown("---")

    st.markdown(
        """
        ### 👥 Team Members

        - Aina
        - Joey
        - Tony
        - Yiting
        """
    )

# ==========================================================
# Main Page
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

pages[selected_page].render()

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "🏦 Mortgage Loan Prediction Dashboard • PODI Final Project • Built with Streamlit"
)