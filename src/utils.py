from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "mortgage_loan_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)