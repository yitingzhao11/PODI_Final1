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

    st.title("🏦 Mortgage Loan Prediction System")

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