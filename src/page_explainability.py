"""
Page 4 — Explainability
Mortgage Loan Prediction Project
=================================

Uses:

✓ SHAP
✓ Model Importance
✓ Permutation Importance
✓ Linear Regression
✓ Random Forest

"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def render():

    st.title("🔍 Explainability Analysis")

    st.caption(
        "Understand which borrower characteristics most strongly influence predicted mortgage loan amounts."
    )

    st.divider()

    st.markdown("""
    <style>

    .stApp{
        background-color:#F8FAFC;
    }

    </style>
    """, unsafe_allow_html=True)

    ############################################################
    # Load Data
    ############################################################

    from src.utils import load_data

    df=load_data()

    target="Max Loan Amount (USD)"

    features=[

    'Gender',
    'Age',
    'Married',
    'Education',
    'Job',
    'Employment Years',
    'Annual Income (USD)',
    'Interest Rate',
    'Down Payment (USD)',
    'Credit Score',
    'Existing Monthly Debt (USD)',
    'Area',
    'Loans Repaid'

    ]

    X=df[features]
    y=df[target]

    ############################################################
    # Detect feature types
    ############################################################

    numeric_features=(
        X.select_dtypes(
            include=['int64','float64']
        )
        .columns
        .tolist()
    )

    categorical_features=(
        X.select_dtypes(
            include=['object']
        )
        .columns
        .tolist()
    )

    ############################################################
    # Preprocessing
    ############################################################

    preprocessor=ColumnTransformer(

    transformers=[

    (
    'cat',
    OneHotEncoder(
    handle_unknown='ignore'
    ),
    categorical_features
    ),

    (
    'num',
    'passthrough',
    numeric_features
    )

    ]

    )

    ############################################################
    # Split data
    ############################################################

    X_train,X_test,y_train,y_test=(
    train_test_split(
    X,
    y,
    test_size=.2,
    random_state=42
    )
    )

    ############################################################
    # Model Choice
    ############################################################

    st.subheader("Model Selection")

    model_choice = st.selectbox(
        "Select a prediction model",
        [
            "Linear Regression",
            "Random Forest"
        ]
    )

    st.info(
    """
    This page explains how the selected machine learning model predicts
    maximum mortgage loan amounts.

    Three complementary explainability techniques are provided:

    • Model Importance\n
    • Permutation Importance\n
    • SHAP Values\n
    """
    )

    ############################################################
    # Train model
    ############################################################

    @st.cache_resource
    def train(model_name):

        if model_name=="Linear Regression":

            model=Pipeline([

                (
                'preprocessor',
                preprocessor
                ),

                (
                'model',
                LinearRegression()
                )

            ])

        else:

            model=Pipeline([

                (
                'preprocessor',
                preprocessor
                ),

                (
                'model',
                RandomForestRegressor(

                    n_estimators=200,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1

                )
                )

            ])

        model.fit(
            X_train,
            y_train
        )

        return model


    model=train(model_choice)

    ############################################################
    # Get transformed features
    ############################################################

    X_train_encoded = (
        model.named_steps["preprocessor"]
        .transform(X_train)
    )

    X_test_encoded=(
    model.named_steps[
    'preprocessor'
    ]
    .transform(
    X_test
    )
    )

    feature_names=(
    model.named_steps[
    'preprocessor'
    ]
    .get_feature_names_out()
    )

    ############################################################
    # Tabs
    ############################################################

    tab1,tab2,tab3=st.tabs([

    "🌲 Model Importance",
    "🔀 Permutation",
    "💎 SHAP"

    ])

    ############################################################
    # Model Importance
    ############################################################

    with tab1:

        st.subheader(
            "Model Importance"
        )

        st.markdown("""
        Built-in feature importance estimates how much each variable contributes
        to predicting the maximum mortgage loan amount.
        """)

        if model_choice=="Random Forest":

            importances=(
            model.named_steps[
            'model'
            ].feature_importances_
            )

        else:

            importances=np.abs(

            model.named_steps[
            'model'
            ].coef_

            )

        importance_df=(
        pd.DataFrame({

        "Feature":feature_names,
        "Importance":importances

        })
        .sort_values(
        "Importance"
        )
        )

        fig=px.bar(

        importance_df.tail(15),

        title = 'Model Importance',

        x='Importance',
        y='Feature',

        orientation='h',

        color='Importance',

        color_continuous_scale='Blues'

        )

        fig.update_layout(
            template="plotly_white",
            height=450,
            title_x=0.5,
            margin=dict(l=20,r=20,t=40,b=20)
        )

        st.plotly_chart(
        fig,
        use_container_width=True
        )

    ############################################################
    # Permutation
    ############################################################

    with tab2:

        st.subheader(
        "Permutation Importance"
        )

        st.markdown("""
        Permutation importance measures how much model performance decreases
        when a feature's values are randomly shuffled.
        Larger decreases indicate more influential variables.
        """)

        perm=(
        permutation_importance(

        model,
        X_test,
        y_test,

        n_repeats=5,
        random_state=42

        )
        )

        perm_df=(
        pd.DataFrame({

        "Feature":features,
        "Importance":
        perm.importances_mean

        })
        .sort_values(
        "Importance"
        )
        )

        fig=px.bar(

        perm_df,

        title = 'Permutation',

        x='Importance',
        y='Feature',

        orientation='h',

        color='Importance',

        color_continuous_scale='Purples'

        )

        fig.update_layout(
            template="plotly_white",
            height=450,
            title_x=0.5,
            margin=dict(l=20,r=20,t=40,b=20)
        )

        st.plotly_chart(
        fig,
        use_container_width=True
        )

    ############################################################
    # SHAP
    ############################################################

    with tab3:

        st.subheader(
        "SHAP Analysis"
        )

        st.markdown("""
        SHAP (SHapley Additive exPlanations) explains how each borrower
        characteristic increases or decreases the predicted loan amount
        for individual observations.
        """)

        if model_choice=="Random Forest":

            explainer=shap.TreeExplainer(

            model.named_steps[
            'model'
            ]
            )

            shap_values=(
            explainer.shap_values(
            X_test_encoded
            )
            )

        else:

            explainer=shap.Explainer(

            model.named_steps[
            'model'
            ],

            X_train_encoded

            )

            shap_values=(
            explainer(
            X_test_encoded
            )
            )

            shap_values=shap_values.values

        mean_shap=(
        np.abs(
        shap_values
        ).mean(axis=0)
        )

        shap_df=(
        pd.DataFrame({

        "Feature":feature_names,
        "SHAP Importance":
        mean_shap

        })
        .sort_values(
        "SHAP Importance"
        )
        )

        fig=px.bar(

        shap_df.tail(15),

        title = "SHAP",

        x='SHAP Importance',
        y='Feature',

        orientation='h',

        color='SHAP Importance',

        color_continuous_scale='Reds'

        )

        fig.update_layout(
            template="plotly_white",
            height=450,
            title_x=0.5,
            margin=dict(l=20,r=20,t=40,b=20)
        )

        st.plotly_chart(
        fig,
        use_container_width=True
        )

    ############################################################
    # Interpretation Section
    ############################################################

    st.markdown("---")

    st.subheader("💡 Interpretation")

    st.markdown("""
    The explainability analysis highlights which borrower characteristics have
    the greatest influence on the predicted maximum mortgage loan amount.

    Typical high-impact features include:

    - Annual Income (USD)
    - Credit Score
    - Down Payment (USD)
    - Existing Monthly Debt (USD)
    - Employment Years

    Across all three methods, features with higher importance values have a
    greater influence on the model's predictions.

    For SHAP analysis:

    - Positive SHAP values increase the predicted loan amount.
    - Negative SHAP values decrease the predicted loan amount.
    """)