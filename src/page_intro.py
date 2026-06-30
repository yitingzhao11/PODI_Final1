"""
Page 1 — Business Case & Data Presentation
============================================
Presents the problem statement, dataset overview,
descriptive statistics, and data quality checks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from src.utils import load_data


##########################################################
# Dataset Overview Function
##########################################################

def render_data_overview(df):

    st.subheader(
        "📊 Dataset Overview"
    )

    missing_count = int(
        df.isna().sum().sum()
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Rows",
        f"{len(df):,}"
    )

    c2.metric(
        "Columns",
        len(df.columns)
    )

    c3.metric(
        "Missing",
        missing_count
    )

    c4.metric(
        "Duplicates",
        duplicate_count
    )

    #######################################################

    st.markdown(
        "### Dataset Preview"
    )

    rows=st.slider(
        "Rows to display",
        5,
        20,
        5
    )

    st.dataframe(
        df.head(rows),
        use_container_width=True,
        hide_index=True
    )


    st.markdown("### Data Quality")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing.to_frame("Missing Values"))

    #######################################################

    st.markdown(
        "### Column Information"
    )

    column_info=pd.DataFrame({

        "Column":
        df.columns,

        "Data Type":
        df.dtypes.astype(str),

        "Missing":
        df.isna().sum(),

        "Missing %":
        (
            df.isna().mean()*100
        ).round(2)

    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    #######################################################

    st.markdown(
        "### Summary Statistics"
    )

    if st.button(
        "Show Statistics"
    ):

        summary = (
            df.describe(include="all")
            .transpose()
            .round(2)
        )

        st.dataframe(
            summary,
            use_container_width=True
        )

    #######################################################

    st.markdown(
        "### Feature Distribution"
    )

    numeric_columns=(

        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()

    )

    if numeric_columns:

        selected=st.selectbox(
            "Select feature",
            numeric_columns
        )

        fig = px.histogram(
            df,
            x=selected,
            nbins=30,
            color_discrete_sequence=["#2563EB"],
            marginal="box"
        )

        fig.update_layout(
            template="plotly_white",
            height=450,
            title=f"Distribution of {selected}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


##########################################################
# Main Render Function
##########################################################

def render():

    ##########################################################
    # Custom CSS
    ##########################################################

    st.markdown("""

    <style>

    .stApp{
        background-color:#F8FAFC;
    }

    .block-container{
        padding-top:2rem;
    }

    .hero-banner{

        background:
        linear-gradient(
        135deg,
        #0F172A 0%,
        #1E3A8A 50%,
        #2563EB 100%
        );

        color:white;
        padding:3rem;
        border-radius:28px;
        margin-bottom:2rem;

        box-shadow:
        0 12px 35px rgba(0,0,0,0.18);

        text-align:center;
    }

    .hero-banner h1{

        font-size:3rem;
        font-weight:800;
        color:white;
    }

    .hero-banner p{

        font-size:1.15rem;
        color:rgba(255,255,255,.92);

    }

    .hero-stats{

        display:flex;
        justify-content:center;
        gap:60px;
        margin-top:2rem;
        flex-wrap:wrap;

    }

    .hero-stat{
        text-align:center;
    }

    .hero-stat h2{

        color:white;
        font-size:2rem;
        margin:0;

    }

    .badge{

        display:inline-block;
        background:rgba(255,255,255,.15);

        color:white;

        border:1px solid rgba(255,255,255,.25);

        padding:10px 18px;

        border-radius:30px;

        margin:5px;

    }

    div[data-testid="stMetric"]{

        background:white;
        border-radius:15px;
        padding:18px;

        box-shadow:
        0 2px 10px rgba(0,0,0,.08);

    }

    </style>

    """,unsafe_allow_html=True)

    st.title("🏦 Mortgage Loan Prediction")

    st.caption(
        "An interactive machine learning application for exploring mortgage loan data, "
        "predicting maximum loan amounts, and interpreting model predictions."
    )

    st.divider()

    image_path = Path(__file__).resolve().parent.parent / "assets" / "mortgage_loan.jpg"


    col1, col2, col3 = st.columns([1, 6, 1])

    with col2:
        st.image(image_path, use_container_width=True)


    st.divider()

    ##########################################################
    # Load Data
    ##########################################################

    try:
        df = load_data()

    except Exception as e:
        st.error(f"Unable to load dataset.\n\n{e}")
        st.stop()

    ##########################################################
    # Business Problem
    ##########################################################

    st.subheader("🎯 Business Problem")

    st.info("""
    Mortgage approval decisions can be time-consuming, difficult to scale, and susceptible to human bias.

    **Project Objective**

    Predict the **Maximum Loan Amount (USD)** using borrower characteristics such as:

    - Annual Income
    - Credit Score
    - Existing Monthly Debt
    - Down Payment
    - Employment History
    - Education Level

    **Business Value**

    - Faster loan approval decisions
    - Improved risk assessment
    - More consistent lending decisions
    - Enhanced customer experience
    """)


    ##########################################################
    # Call overview section
    ##########################################################
    st.divider()
    render_data_overview(df)