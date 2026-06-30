import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from src.utils import load_data

def render():
# ==========================================================
# Page Title
# ==========================================================
    st.title("🏠 Mortgage Loan Prediction")
    st.markdown("""
    Predict the **Maximum Loan Amount (USD)** for a borrower using Machine Learning.

    ### Available Models
    - Linear Regression
    - Random Forest Regressor
    """)

    # ==========================================================
    # Load Dataset
    # ==========================================================

    df = load_data()

    target = "Max Loan Amount (USD)"

    features = [
        "Gender",
        "Age",
        "Married",
        "Education",
        "Job",
        "Employment Years",
        "Annual Income (USD)",
        "Interest Rate",
        "Down Payment (USD)",
        "Credit Score",
        "Existing Monthly Debt (USD)",
        "Area",
        "Loans Repaid"
    ]

    features = [col for col in features if col in df.columns]

    model_df = df[features + [target]].dropna()

    X = model_df[features]
    y = model_df[target]

    # ==========================================================
    # Determine Data Types
    # ==========================================================

    numeric_features = (
        X.select_dtypes(include=["int64", "float64"])
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(include=["object", "category", "bool"])
        .columns
        .tolist()
    )

    # ==========================================================
    # Preprocessing
    # ==========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # ==========================================================
    # Train Models
    # ==========================================================

    @st.cache_resource
    def train_models():

        linear_model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        )

        rf_model = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        linear_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

        return linear_model, rf_model


    linear_model, rf_model = train_models()

    # ==========================================================
    # Evaluate Models
    # ==========================================================

    def evaluate(model):
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        return mae, rmse, r2


    linear_mae, linear_rmse, linear_r2 = evaluate(linear_model)
    rf_mae, rf_rmse, rf_r2 = evaluate(rf_model)

    # ==========================================================
    # Performance Section
    # ==========================================================

    st.header("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Linear Regression")
        st.metric("R² Score", f"{linear_r2:.3f}")
        st.metric("RMSE", f"${linear_rmse:,.0f}")

    with col2:
        st.subheader("Random Forest")
        st.metric("R² Score", f"{rf_r2:.3f}")
        st.metric("RMSE", f"${rf_rmse:,.0f}")

    # ==========================================================
    # Borrower Inputs
    # ==========================================================

    st.header("📝 Enter Borrower Information")

    input_data = {}

    left, right = st.columns(2)

    for i, feature in enumerate(features):

        current_col = left if i % 2 == 0 else right

        with current_col:

            if feature in numeric_features:

                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())

                input_data[feature]=st.number_input(
                                            feature,
                                            min_value=min_val,
                                            max_value=max_val,
                                            value=mean_val
                                        )

            else:

                options = sorted(
                    df[feature].dropna().unique().tolist()
                )

                input_data[feature] = st.selectbox(
                    feature,
                    options,
                )

    input_df = pd.DataFrame([input_data])

    # ==========================================================
    # Model Selection
    # ==========================================================

    st.header("🤖 Choose a Model")

    model_choice = st.radio(
        "",
        ["Linear Regression", "Random Forest"],
        horizontal=True,
    )

    # ==========================================================
    # Prediction Button
    # ==========================================================

    if st.button("Predict Maximum Loan Amount"):

        if model_choice == "Linear Regression":
            prediction = linear_model.predict(input_df)[0]
        else:
            prediction = rf_model.predict(input_df)[0]

        st.success(
            f"Predicted Maximum Loan Amount: "
            f"${prediction:,.2f}"
        )

        # Borrowing category
        if prediction < 200000:
            category = "🟢 Low Loan Amount"
        elif prediction < 500000:
            category = "🟡 Medium Loan Amount"
        else:
            category = "🔴 High Loan Amount"

        st.info(f"Borrowing Category: {category}")

    # ==========================================================
    # Compare Models
    # ==========================================================

    st.header("🔍 Model Comparison")

    linear_pred = linear_model.predict(input_df)[0]
    rf_pred = rf_model.predict(input_df)[0]

    comparison = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Random Forest",
            ],
            "Predicted Loan Amount (USD)": [
                linear_pred,
                rf_pred,
            ],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
    )

    # ==========================================================
    # Borrower Summary
    # ==========================================================

    with st.expander("View Borrower Profile"):
        st.dataframe(input_df)

    # ==========================================================
    # Dataset Preview
    # ==========================================================

    with st.expander("View Dataset"):
        st.dataframe(df.head())