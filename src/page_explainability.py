"""
Page 4 — Explainability
Mortgage Loan Prediction Project
=================================

Uses:

✓ SHAP
✓ Feature Importance
✓ Permutation Importance
✓ Linear Regression
✓ Random Forest

"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def render():
    ############################################################
    # Styling
    ############################################################

    st.markdown("""
    <style>

    .stApp{
    background-color:#F8FAFC;
    }

    .main-title{
    background:linear-gradient(
    135deg,
    #0F172A,
    #1E3A8A,
    #2563EB
    );

    padding:2rem;
    border-radius:25px;
    color:white;
    text-align:center;
    margin-bottom:2rem;
    box-shadow:0 10px 30px rgba(0,0,0,0.15);
    }

    .card{
    background:white;
    padding:20px;
    border-radius:18px;
    box-shadow:0 5px 15px rgba(0,0,0,.08);
    }

    </style>
    """,unsafe_allow_html=True)


    st.markdown("""
    <div class='main-title'>

    <h1>🔍 Explainable AI Dashboard</h1>

    <p>
    Understand what drives mortgage loan predictions and discover
    the most important borrower characteristics.
    </p>

    </div>
    """,unsafe_allow_html=True)

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

    st.subheader(
    "Choose Model"
    )

    model_choice=st.selectbox(

    "",
    [
    "Linear Regression",
    "Random Forest"
    ]

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

    X_train_encoded=(
    model.named_steps[
    'preprocessor'
    ]
    .fit_transform(
    X_train
    )
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

    "🌲 Feature Importance",
    "🔀 Permutation",
    "💎 SHAP"

    ])

    ############################################################
    # Feature Importance
    ############################################################

    with tab1:

        st.subheader(
            "Feature Importance"
        )

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

        x='Importance',
        y='Feature',

        orientation='h',

        color='Importance',

        color_continuous_scale='Blues'

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

        x='Importance',
        y='Feature',

        orientation='h',

        color='Importance',

        color_continuous_scale='Purples'

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

        st.info(
        """
        SHAP explains how each borrower characteristic
        contributes to the final loan prediction.
        Positive values increase prediction.
        Negative values decrease prediction.
        """
        )

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

        x='SHAP Importance',
        y='Feature',

        orientation='h',

        color='SHAP Importance',

        color_continuous_scale='Reds'

        )

        st.plotly_chart(
        fig,
        use_container_width=True
        )

    ############################################################
    # Interpretation Section
    ############################################################

    st.markdown("---")

    st.markdown(
    """
    ### 💡 Key Interpretation

    Features with larger values have stronger influence on loan prediction.

    Typical drivers:

    ✅ Annual Income  
    ✅ Credit Score  
    ✅ Down Payment  
    ✅ Existing Debt  
    ✅ Employment Years  

    Positive SHAP values push predicted loan amount upward.

    Negative SHAP values reduce predicted loan amount.
    """
    )