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

    st.write(
        """
        The mortgage dataset contains applicant demographic, financial,
        employment, and loan-related information that can be used to
        predict the maximum eligible mortgage loan amount.

        During exploratory data analysis, numerical variables such as
        applicant income, loan amount, credit score, interest rate,
        employment history, and debt were examined to understand their
        distributions and identify potential outliers.

        Categorical variables including education level, marital status,
        occupation, gender, and residential area were prepared for
        machine learning using appropriate encoding techniques.
        """
    )

    st.divider()

    ##########################################################
    # Feature Importance
    ##########################################################

    st.header("⭐ Key Features Influencing Mortgage Predictions")

    st.markdown(
        """
### 1. Credit Score

Credit score is the strongest predictor of mortgage eligibility because it reflects an applicant's history of managing debt. Applicants with stronger credit profiles generally qualify for higher loan amounts and present lower lending risk.

---

### 2. Annual Income

Applicant income measures the ability to repay a mortgage. Higher annual income typically increases the maximum loan amount because lenders consider these applicants financially stronger.

---

### 3. Maximum Loan Amount

Requested loan amount directly affects lending risk. Larger loan requests generally require stronger financial qualifications before approval.

---

### 4. Employment Stability

Employment history provides evidence of consistent income. Applicants with longer employment histories are usually viewed as more financially stable and reliable borrowers.

---

### 5. Existing Monthly Debt

Debt obligations relative to income help lenders evaluate financial burden. Lower debt levels generally improve borrowing capacity and reduce lending risk.

---

### 6. Property Characteristics

Property-related information contributes additional context by representing the value of the loan collateral. Lower-risk properties may improve lending confidence.

---

### 7. Applicant Demographics

Features such as age, education, marital status, gender, and residential area contribute supporting information but generally have less influence than financial variables.
"""
    )

    st.divider()

    ##########################################################
    # Overall Findings
    ##########################################################

    st.header("📈 Overall Findings")

    st.info(
        """
The exploratory analysis indicates that **financial variables are the primary drivers** of mortgage predictions.

Among all features, **Credit Score**, **Annual Income**, and **Maximum Loan Amount** contribute most significantly because they directly represent an applicant's repayment ability and financial risk.

Employment-related variables strengthen prediction accuracy by reflecting income stability, while property characteristics provide additional information regarding loan security.
"""
    )

    st.divider()

    ##########################################################
    # Final Conclusion
    ##########################################################

    st.header("✅ Project Conclusion")

    st.success(
        """
This application demonstrates how machine learning can assist in evaluating mortgage applications using applicant financial, employment, demographic, and loan-related information.

Users can explore the dataset, visualize important trends, generate mortgage loan predictions, and understand model behavior through explainable AI techniques.

Overall, the analysis shows that **Credit Score**, **Annual Income**, and **Existing Financial Obligations** are the strongest factors influencing mortgage decisions, while employment stability and property information further improve prediction performance.

The application provides lenders and analysts with an interactive decision-support tool that improves consistency, increases efficiency, and promotes greater transparency in mortgage lending decisions.
"""
    )