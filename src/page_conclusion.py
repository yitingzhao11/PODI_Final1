import streamlit as st


def render():

    st.title("📄 Project Conclusion")

    st.caption(
        "Summary of the mortgage loan analysis, feature importance, "
        "and the overall project findings."
    )

    st.divider()

    ##########################################################
    # Dataset Summary
    ##########################################################

    st.header("📊 Dataset Analysis")

    st.markdown(
        """
The mortgage dataset contains applicant demographic, financial,
employment, property, and loan-related information used to predict the
maximum eligible mortgage loan amount.

During exploratory data analysis, numerical variables such as applicant
income, credit score, employment history, interest rate, debt, and loan
amount were examined to identify trends and potential outliers.
Categorical variables including education, marital status, occupation,
gender, and residential area were encoded to prepare the data for
machine learning models.
"""
    )

    st.divider()

    ##########################################################
    # Feature Importance
    ##########################################################

    st.header("⭐ Key Features Influencing Mortgage Predictions")

    st.markdown(
        """
### 1. Credit History / Credit Score

Credit history is the strongest predictor because it reflects an applicant's ability to repay previous debts. Applicants with stronger credit profiles are generally considered financially reliable and are more likely to qualify for larger mortgage loans.

---

### 2. Applicant Income

Applicant income measures repayment capacity. Higher income generally increases borrowing power because lenders view these applicants as lower financial risk.

---

### 3. Loan Amount

The requested loan amount directly affects lending decisions. Larger loan requests typically require stronger financial qualifications and therefore carry greater lending risk.

---

### 4. Employment Stability

Employment status and employment history demonstrate consistent income. Applicants with stable, long-term employment are generally viewed as more reliable borrowers.

---

### 5. Supporting Features

Existing debt, property characteristics, education, marital status, gender, and residential area provide additional information that helps improve overall prediction accuracy.
"""
    )

    st.divider()

    ##########################################################
    # Overall Findings
    ##########################################################

    st.header("📈 Overall Findings")

    st.markdown(
        """
The analysis shows that **financial factors** have the greatest influence
on mortgage predictions. Among all variables, **Credit History**,
**Applicant Income**, and **Loan Amount** contribute most significantly
because they directly represent financial reliability and repayment
ability.

Employment stability, existing debt, and property-related features
provide additional context that improves prediction accuracy and enables
the models to generate more reliable mortgage loan estimates.
"""
    )

    st.divider()

    ##########################################################
    # Project Conclusion
    ##########################################################

    st.header("✅ Project Conclusion")

    st.markdown(
        """
This application demonstrates how machine learning can support mortgage
lending decisions by predicting the maximum eligible mortgage loan
amount using applicant financial, employment, demographic, and
property-related information.

Users can explore the dataset, visualize important trends, compare
multiple regression models, optimize model performance through
hyperparameter tuning, generate mortgage predictions, and understand
model behavior using explainable AI techniques.

Overall, **Credit History**, **Applicant Income**, and **Loan Amount**
are the strongest factors influencing mortgage predictions because they
directly reflect repayment ability and financial risk. Employment
stability, existing debt, and property characteristics further improve
prediction performance by providing additional borrower context.

Hyperparameter tuning is performed using **Optuna**, which automatically
searches for the optimal combination of model parameters to improve
prediction accuracy while reducing the need for manual trial-and-error.
"""
    )