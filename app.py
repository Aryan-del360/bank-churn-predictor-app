import streamlit as st
import joblib
import pandas as pd
import numpy as np
# No need to import StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline directly
# if your saved model is a Pipeline object containing them.
# The 'model_pipeline' object will handle these internally upon prediction.

# --- Configuration ---
# Define the exact columns your model expects as input *after* feature engineering
# and *before* the ColumnTransformer in your Kaggle Notebook.
# This order is CRUCIAL for the ColumnTransformer inside your pipeline.
# Adjust this list if your feature engineering or original column order was different.
COLUMNS_ORDER = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'BalanceSalaryRatio', 'IsZeroBalanceAndHasProducts', 'ActiveHasCrCard',
    'Geography_France', 'Geography_Germany', 'Geography_Spain', # One-hot encoded geographies
    'Gender_Female', 'Gender_Male' # One-hot encoded genders
]

# --- Helper Functions ---

@st.cache_resource # Cache the model loading to improve performance
def load_churn_model(model_path='churn_prediction_model.pkl'):
    """Loads the pre-trained churn prediction model pipeline."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app if the model isn't found
    except Exception as e:
        st.error(f"Error loading model: {e}. Please check the model file.")
        st.stop()

def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the feature engineering steps applied during model training.
    IMPORTANT: This function MUST EXACTLY MATCH the feature engineering
    performed in your Kaggle Notebook.
    """
    # Ensure 'EstimatedSalary' is not zero to avoid division by zero
    df['BalanceSalaryRatio'] = df.apply(
        lambda row: row['Balance'] / row['EstimatedSalary'] if row['EstimatedSalary'] > 0 else 0, axis=1
    )
    df['IsZeroBalanceAndHasProducts'] = ((df['Balance'] == 0) & (df['NumOfProducts'] > 0)).astype(int)
    df['ActiveHasCrCard'] = ((df['IsActiveMember'] == 1) & (df['HasCrCard'] == 1)).astype(int)
    return df

def get_user_inputs():
    """Collects user inputs from Streamlit widgets."""
    st.sidebar.header("Customer Details")

    # Using columns for a cleaner layout in the sidebar
    with st.sidebar.expander("Demographics & Profile", expanded=True):
        credit_score = st.slider("Credit Score", 350, 850, 650, help="Score indicating creditworthiness.")
        geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'], help="Customer's country of residence.")
        gender = st.selectbox("Gender", ['Female', 'Male'], help="Customer's gender.")

    with st.sidebar.expander("Account Details", expanded=True):
        age = st.number_input("Age", 18, 92, 38, help="Customer's age in years.")
        tenure = st.slider("Tenure (Years)", 0, 10, 5, help="Number of years customer has been with the bank.")
        num_products = st.slider("Number of Products", 1, 4, 1, help="Number of bank products (e.g., accounts, loans) customer uses.")

    with st.sidebar.expander("Financials & Activity", expanded=True):
        balance = st.number_input("Balance (USD)", 0.0, 250000.0, 70000.0, step=100.0, help="Customer's account balance.")
        estimated_salary = st.number_input("Estimated Salary (USD)", 0.0, 200000.0, 100000.0, step=100.0, help="Estimated annual salary of the customer.")
        has_cr_card = st.checkbox("Has Credit Card?", value=True, help="Check if the customer has a credit card with the bank.")
        is_active_member = st.checkbox("Is Active Member?", value=True, help="Check if the customer is an active member of the bank.")

    # Convert boolean checkboxes to int (0 or 1) as expected by the model
    has_cr_card_val = 1 if has_cr_card else 0
    is_active_member_val = 1 if is_active_member else 0

    input_data_raw = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_cr_card_val,
        'IsActiveMember': is_active_member_val,
        'EstimatedSalary': estimated_salary
    }
    return pd.DataFrame([input_data_raw])

def preprocess_and_predict(model_pipeline, raw_input_df: pd.DataFrame):
    """
    Applies feature engineering and makes a prediction using the loaded model pipeline.
    """
    # Apply feature engineering
    engineered_input_df = create_engineered_features(raw_input_df.copy())

    # Create one-hot encoded columns for 'Geography' and 'Gender'
    # This manually creates the columns expected by the ColumnTransformer after OneHotEncoder
    # Ensure these match the categories and column naming convention from your training
    # (e.g., 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender_Female', 'Gender_Male')
    # If your training data had other categories or different naming, adjust here.

    # Initialize all possible one-hot encoded columns to 0
    for geo in ['France', 'Germany', 'Spain']:
        engineered_input_df[f'Geography_{geo}'] = 0
    for gen in ['Female', 'Male']:
        engineered_input_df[f'Gender_{gen}'] = 0

    # Set the value to 1 for the selected category
    engineered_input_df[f'Geography_{raw_input_df["Geography"].iloc[0]}'] = 1
    engineered_input_df[f'Gender_{raw_input_df["Gender"].iloc[0]}'] = 1

    # Drop original categorical columns as they will be handled by the one-hot encoded versions
    engineered_input_df = engineered_input_df.drop(columns=['Geography', 'Gender'])


    # Ensure the final input DataFrame has the exact same columns and order as the
    # X DataFrame used for training the model (before ColumnTransformer).
    # This is CRUCIAL for the ColumnTransformer inside the pipeline.
    try:
        final_input_df = engineered_input_df[COLUMNS_ORDER]
    except KeyError as e:
        st.error(f"Column mismatch: A required column for prediction is missing or misnamed. Check COLUMNS_ORDER and feature engineering. Missing: {e}")
        st.stop()


    # Make prediction using the loaded pipeline
    # The pipeline automatically handles scaling and the one-hot encoding for the original categorical columns
    prediction_proba = model_pipeline.predict_proba(final_input_df)[:, 1][0]
    prediction = model_pipeline.predict(final_input_df)[0]

    return prediction, prediction_proba

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Bank Customer Churn Predictor", layout="wide", initial_sidebar_state="expanded")

    st.title("🏦 Bank Customer Churn Prediction App")
    st.markdown("---")

    st.write("Welcome! This application helps predict whether a bank customer is likely to churn based on their profile and activity.")
    st.write("Please adjust the customer details in the sidebar and click 'Predict Churn' to see the outcome.")

    # Load the model only once
    model_pipeline = load_churn_model()

    # Get user inputs from the sidebar
    raw_input_df = get_user_inputs()

    st.markdown("---")

    # Prediction button
    if st.button("Predict Churn Likelihood", help="Click to get the churn prediction for the entered customer details."):
        with st.spinner("Predicting..."):
            prediction, prediction_proba = preprocess_and_predict(model_pipeline, raw_input_df)

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"**Customer is Predicted to CHURN!** 😟")
            st.write(f"Probability of Churn: **{prediction_proba:.2f}**")
            st.markdown("---")
            st.info("💡 **Actionable Insight:** This customer shows a high likelihood of churning. Consider targeted retention strategies such as personalized offers, loyalty programs, or proactive customer service outreach to prevent them from leaving.")
        else:
            st.success(f"**Customer is Predicted to STAY!** 🎉")
            st.write(f"Probability of Churn: **{prediction_proba:.2f}**") # Still show probability even if low
            st.markdown("---")
            st.info("💡 **Actionable Insight:** This customer is likely to remain active. Continue to monitor engagement and ensure their satisfaction with bank services.")

    st.markdown("---")
    st.markdown("### About This Predictor")
    st.info("""
        This application uses a pre-trained Machine Learning model (an XGBoost Classifier within a scikit-learn Pipeline)
        to predict customer churn. The model was trained on historical bank customer data, learning patterns that indicate
        whether a customer is likely to leave.

        Your inputs are processed through the same steps used during model training, including:
        - **Feature Engineering:** Creating new, insightful features from raw data.
        - **Scaling:** Normalizing numerical features.
        - **One-Hot Encoding:** Converting categorical data into a numerical format.

        This ensures the model receives data in the exact format it expects for accurate predictions.
        """)
    st.markdown("---")
    st.caption("Developed by Shubham Sharma as a Data Science Portfolio Project.")

if __name__ == '__main__':
    main()
