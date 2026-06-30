"""
Page 5 - Hyperparameter Tuning
================================
Models:
- Ridge
- Random Forest
- Gradient Boosting
"""

## import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
import optuna
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import wandb
from src import wandb_tracker
from datetime import datetime
import os



RANDOM_STATE = 42


def render():
    st.title("⚙️ Mortgage Loan Hyperparameter Tuning")

    st.caption(
        """
        Hyperparameter tuning compares multiple regression models and select the best performing one.
        Experiments can also be tracked using Weights & Biases.
        """
    )

    st.divider()

########################################################
# LOAD DATASET
########################################################

    df = pd.read_csv("mortgage_loan_dataset.csv")

    target = "Max Loan Amount (USD)"

    ## Drop target only
    drop_cols = [target]

    features = [
        c for c in df.columns
        if c not in drop_cols
    ]

##########################################################
# Data Preprocessing
##########################################################

    X = pd.get_dummies(
        df[features],
        drop_first = True
    )

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = RANDOM_STATE
    )

##########################################################
# Configuration/Interactive Features
##########################################################

    st.subheader("Configuration")

    c1, c2, c3 = st.columns(3)

    with c1:
        model_name = st.selectbox(
            "Choose model",
            ["Ridge", "Random Forest", "Gradient Boosting"]
        )

    with c2:
        number_trials = st.slider(
            "Number of trials",
            min_value = 5,
            max_value = 50,
            value = 15
        )

    with c3:
        cv = st.slider(
            "Cross-Validation folds",
            min_value = 3,
            max_value = 10,
            value = 5
        )

##########################################################
# Search Space
##########################################################

    search_space = {

        "Ridge":
        {
            "alpha": "0.001-100"
        },

        "Random Forest":
        {
            "n_estimators": "50-300",
            "max_depth": "3-20",
            "min_samples_split": "2-10",
            "min_samples_leaf": "1-5"
        },

        "Gradient Boosting":
        {
            "n_estimators": "50-300",
            "learning_rate": "0.01-0.3",
            "max_depth": "2-10"
        }
    }

    st.subheader("🔎 Hyperparameter Search Space")

    space_df = pd.DataFrame(
        [
            {"Parameter": k, "Range": v}
            for k, v in search_space[model_name].items()
        ]
    )

    st.dataframe(
        space_df,
        use_container_width=True
    )

##########################################################
# W&B
##########################################################

    track_wandb = st.checkbox(
        "💡 Track model performances in Weights & Biases",
        value = wandb_tracker.is_available(),
        disabled = not wandb_tracker.is_available(),
        help = "Set WANDB_API_KEY in .env to enable.",
    )

##########################################################
# Run optimization
##########################################################

    if st.button(
        "🚀 Start Optimization",
        use_container_width=True
    ):

        try:
            optuna.logging.set_verbosity(
                optuna.logging.WARNING
            )

        except ImportError:
            st.error("Install Optuna: `pip install optuna`")
            st.stop()

        wb_run = None

        if track_wandb:
            wb_run = wandb_tracker.init_run(
                run_name = (
                    f"{model_name}_"
                    f"{datetime.now():%Y%m%d_%H%M%S}"
                ),
                config = {
                    "model": model_name,
                    "trials": number_trials,
                    "cv": cv
                }
            )

            if wb_run:
                st.success("Weights & Biases tracking enabled!")
                st.write(f"Run URL: {wb_run.url}")

        def objective(trial):

            if model_name == "Ridge":

                alpha = trial.suggest_float(
                    "alpha",
                    0.001,
                    100,
                    log=True
                )

                model = Pipeline([
                    (
                        "scaler",
                        StandardScaler()
                    ),

                    (
                        "model",
                        Ridge(alpha=alpha)
                    )
                ])

            elif model_name == "Random Forest":

                params = {

                    "n_estimators":
                    trial.suggest_int(
                        "n_estimators",
                        50,
                        300
                    ),

                    "max_depth":
                    trial.suggest_int(
                        "max_depth",
                        3,
                        20
                    ),

                    "min_samples_split":
                    trial.suggest_int(
                        "min_samples_split",
                        2,
                        10
                    ),

                    "min_samples_leaf":
                    trial.suggest_int(
                        "min_samples_leaf",
                        1,
                        5
                    ),

                    "random_state":
                    RANDOM_STATE
                }

                model = RandomForestRegressor(
                    **params
                )

            else:

                params = {

                    "n_estimators":
                    trial.suggest_int(
                        "n_estimators",
                        50,
                        300
                    ),

                    "learning_rate":
                    trial.suggest_float(
                        "learning_rate",
                        0.01,
                        0.3
                    ),

                    "max_depth":
                    trial.suggest_int(
                        "max_depth",
                        2,
                        10
                    ),

                    "random_state":
                    RANDOM_STATE
                }

                model = GradientBoostingRegressor(
                    **params
                )

            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="r2"
            )

            score = scores.mean()

            if wb_run:
                wandb_tracker.log_metrics(
                    wb_run,
                    {
                        "trial": trial.number,
                        "cv_r2": score,
                        **trial.params,
                        "Test R2": r2,
                        "Test MAE": mae,
                        "Test RMSE": rmse,
                        "Best CV R2": study.best_value

                    },
                    step=trial.number
                )

            return score

        progress = st.progress(0)

        study = optuna.create_study(
            direction = "maximize"
        )

        def callback(study, trial):

            progress.progress(
                (trial.number + 1) / number_trials
            )

        study.optimize(
            objective,
            n_trials=number_trials,
            callbacks=[callback]
        )

##########################################################
# Best Performing Model
##########################################################

        best_params = study.best_params

        if wb_run:
            wb_run.config.update(
                {"best_params": best_params},
                allow_val_change=True
            )

        if model_name == "Ridge":

            best_model = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "model",
                    Ridge(**best_params)
                )
            ])

        elif model_name == "Random Forest":

            best_model = RandomForestRegressor(
                **best_params,
                random_state=RANDOM_STATE
            )

        else:

            best_model = GradientBoostingRegressor(
                **best_params,
                random_state=RANDOM_STATE
            )

        best_model.fit(
            X_train,
            y_train
        )

        pred = best_model.predict(
            X_test
        )

##########################################################
# Metrics
##########################################################

        r2 = r2_score(
            y_test,
            pred
        )

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        st.subheader("🎉 Best Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "R^2",
            f"{r2:.3f}"
        )

        c2.metric(
            "MAE",
            f"{mae:.2f}"
        )

        c3.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        st.write("Best Parameters:")
        st.json(best_params)

##########################################################
# Optimization History
##########################################################

        trials_df = study.trials_dataframe()

        if wb_run:
            table = wandb.Table(dataframe=trials_df)
            wb_run.log({"Optimization History": table})

        fig = px.line(
            trials_df,
            x = "number",
            y = "value",
            title = "Optimization Progress"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

##########################################################
# Actual vs Predicted
##########################################################

        fig2 = px.scatter(
            x = y_test,
            y = pred,
            labels = {
                "x": "Actual",
                "y": "Predicted"
            },

            title="Actual vs Predicted"
        )

        fig2.add_trace(
            go.Scatter(
                x = [
                    y_test.min(),
                    y_test.max()
                ],

                y = [
                    y_test.min(),
                    y_test.max()
                ],

                mode = "lines"
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

##########################################################
# W&B Logging
##########################################################

        if wb_run:

            wandb_tracker.log_metrics(
                wb_run,
                {
                    "R2": r2,
                    "MAE": mae,
                    "RMSE": rmse
                }
            )
        
        if wb_run:
            wb_run.log({
                "Optimization Progress": fig,
                "Actual vs Predicted": fig2
            })

        wandb_tracker.finish_run(
            wb_run
        )


    st.write("W&B Available:", wandb_tracker.is_available())
    st.write("API Key Loaded:", bool(os.environ.get("WANDB_API_KEY")))
    st.write("Project:", os.environ.get("WANDB_PROJECT"))

    key = os.environ.get("WANDB_API_KEY", "")
    st.write("Key starts with:", key[:10])
    st.write("Length:", len(key))