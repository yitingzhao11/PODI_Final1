import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns


##########################################################
# Load Dataset
##########################################################

from src.utils import load_data

##########################################################
# Cache Dataset
##########################################################

@st.cache_data(show_spinner="Loading mortgage data...")
def cached_load_data():
    return load_data()


##########################################################
# Sidebar Filters
##########################################################

def filter_data(data):

    st.sidebar.header("Filters")

    st.sidebar.caption(
        "Selections are combined. Clear a filter to include all values."
    )

    gender_options = sorted(
        data["Gender"]
        .dropna()
        .unique()
    )

    selected_gender = st.sidebar.multiselect(
        "Gender",
        options=gender_options,
        default=gender_options
    )

    married_options = sorted(
        data["Married"]
        .dropna()
        .unique()
)

    selected_married = st.sidebar.multiselect(
        "Marital Status",
        options=married_options,
        default=married_options
    )

    area_options = sorted(
        data["Area"]
        .dropna()
        .unique()
    )

    selected_area = st.sidebar.multiselect(
        "Area",
        options=area_options,
        default=area_options
    )

    education_options = sorted(
        data["Education"]
        .dropna()
        .unique()
    )

    selected_education = st.sidebar.multiselect(
        "Education",
        options=education_options,
        default=education_options
    )

    job_options = sorted(
        data["Job"]
        .dropna()
        .unique()
    )

    selected_job = st.sidebar.multiselect(
        "Job",
        options=job_options,
        default=job_options
    )

    filtered = data.loc[
        (data["Gender"].isin(selected_gender))
        &
        (data["Married"].isin(selected_married))
        &
        (data["Area"].isin(selected_area))
        &
        (data["Education"].isin(selected_education))
        &
        (data["Job"].isin(selected_job))
    ]

    st.sidebar.divider()

    st.sidebar.caption(
        f"Source records: {len(data):,}"
    )

    return filtered


##########################################################
# KPI Cards
##########################################################

def render_kpis(data):

    total_applications = len(data)

    avg_income = data[
        "Annual Income (USD)"
    ].mean()

    avg_credit = data[
        "Credit Score"
    ].mean()

    avg_loan = data[
        "Max Loan Amount (USD)"
    ].mean()

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Applications",
        f"{total_applications:,}"
    )

    col2.metric(
        "Average Income",
        f"${avg_income:,.0f}"
    )

    col3.metric(
        "Average Credit Score",
        f"{avg_credit:.0f}"
    )

    col4.metric(
        "Average Loan Amount",
        f"${avg_loan:,.0f}"
    )

    st.divider()


##########################################################
# Bar Chart
##########################################################

def render_bar_chart(data):

    st.markdown("## 📊 Average Metrics by Category")

    st.markdown(
    """
    Compare the average value of different financial metrics across borrower groups.
    Choose both the grouping category and the metric below.
    """
    )

    grouping_options = {

        "Gender":
        "Gender",

        "Education":
        "Education",

        "Area":
        "Area",

        "Job":
        "Job"
    }

    metric_options = {

        "Annual Income":
        "Annual Income (USD)",

        "Credit Score":
        "Credit Score",

        "Max Loan Amount":
        "Max Loan Amount (USD)",

        "Existing Debt":
        "Existing Monthly Debt (USD)"
    }

    c1,c2 = st.columns(2)

    group_label = c1.selectbox(
        "Group by",
        list(grouping_options)
    )

    metric_label = c2.selectbox(
        "Metric",
        list(metric_options)
    )

    group_column = grouping_options[
        group_label
    ]

    metric_column = metric_options[
        metric_label
    ]

    chart_data = (

        data.groupby(
            group_column,
            as_index=False
        )[metric_column]
        .mean()
    )

    fig = px.bar(

        chart_data,

        x=group_column,

        y=metric_column,

        color=group_column,

        title=f"{metric_label} by {group_label}"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        font=dict(size=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.divider()


##########################################################
# Line Chart
##########################################################

def render_line_chart(data):

    st.markdown("## 📈 Trends Across Age Groups")

    st.markdown(
    """
    Observe how the selected financial metric changes across applicant ages.
    This chart highlights overall trends rather than individual observations.
    """
    )

    metric_options = {

        "Annual Income":
        "Annual Income (USD)",

        "Credit Score":
        "Credit Score",

        "Max Loan Amount":
        "Max Loan Amount (USD)"
    }

    metric_label = st.selectbox(
        "Line chart metric",
        list(metric_options)
    )

    metric_column = metric_options[
        metric_label
    ]

    age_data = (

        data.groupby(
            "Age"
        )[metric_column]
        .mean()
        .reset_index()

    )

    fig = px.line(

        age_data,

        x="Age",

        y=metric_column,

        title=f"{metric_label} Across Age"

    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        font=dict(size=15),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.divider()



##########################################################
# Correlation Heatmap
##########################################################

def render_heatmap(data):

    st.subheader(
        "🔥 Correlation Between Financial Variables"
    )

    st.markdown(
    """
    The heatmap summarizes correlations among numeric variables.

    - Values close to **+1** indicate strong positive relationships.
    - Values close to **−1** indicate strong negative relationships.
    - Values near **0** indicate weak linear relationships.
    """
    )

    numeric_columns = [

        "Age",
        "Employment Years",
        "Annual Income (USD)",
        "Interest Rate",
        "Down Payment (USD)",
        "Credit Score",
        "Existing Monthly Debt (USD)",
        "Loans Repaid",
        "Max Loan Amount (USD)"
    ]

    available_columns = [

        col
        for col in numeric_columns
        if col in data.columns
    ]

    corr = (
        data[
            available_columns
        ]
        .corr(
            numeric_only=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(12,7)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=ax
    )

    st.pyplot(
        fig
    )

    plt.close(fig)


##########################################################
# Main
##########################################################

def render():

    st.title("📊 Data Visualization Dashboard")

    st.caption(
        "Explore borrower characteristics, financial trends, and mortgage loan patterns. "
        "Use the sidebar filters to focus the analysis—every KPI and visualization updates automatically."
    )

    try:

        data = cached_load_data()

    except Exception as e:

        st.error(
            f"Unable to load dataset.\n\n{e}"
        )

        st.stop()

    filtered = filter_data(
        data
    )

    st.caption(
        f"Showing {len(filtered):,} of {len(data):,} applicant records"
    )

    if filtered.empty:

        st.warning(
            "No applicants match this filter combination. Adjust the sidebar filters."
        )

        return

    render_kpis(
        filtered
    )

    c1,c2 = st.columns(2)

    with c1:
        render_bar_chart(
            filtered
        )

    with c2:
        render_line_chart(
            filtered
        )

    render_heatmap(
        filtered
    )

    with st.expander(
        "View Filtered Dataset"
    ):

        rows = st.slider(
            "Rows to display",
            min_value=10,
            max_value=500,
            value=50
        )

        st.dataframe(
            filtered.head(rows),
            use_container_width=True,
            hide_index=True
        )