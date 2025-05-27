import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Define the exact columns your model's ColumnTransformer expects as input
# These should be the original numerical, original categorical, and engineered features.
# The ColumnTransformer itself will handle the one-hot encoding of 'Geography' and 'Gender'.
COLUMNS_ORDER = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'BalanceSalaryRatio', 'IsZeroBalanceAndHasProducts', 'ActiveHasCrCard',
    'Geography', # Keep original categorical column
    'Gender'     # Keep original categorical column
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

@st.cache_data # Cache the training data loading
def load_training_data(data_path='train.csv'):
    """Loads the training data for insights."""
    try:
        df = pd.read_csv(data_path)
        # Apply the same feature engineering to the training data for insights
        df = create_engineered_features(df.copy()) # Apply FE here for consistency
        return df
    except FileNotFoundError:
        st.warning(f"Warning: Training data file '{data_path}' not found. Visual insights from training data will not be available.")
        return None
    except Exception as e:
        st.warning(f"Warning: Error loading training data: {e}. Visual insights from training data might be limited.")
        return None

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

    with st.sidebar.expander("Demographics & Profile", expanded=True):
        credit_score = st.slider("Credit Score", 350, 850, 650, help="Score indicating creditworthiness.")
        geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'], help="Customer's country of residence.")
        gender = st.selectbox("Gender", ['Female', 'Male'], help="Customer's gender.")

    with st.sidebar.expander("Account Details", expanded=True):
        age = st.number_input("Age", 18, 92, 38, help="Customer's age in years.")
        tenure = st.slider("Tenure (Years)", 0, 10, 5, help="Number of bank products (e.g., accounts, loans) customer uses.")
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
    The pipeline handles all further preprocessing (scaling, one-hot encoding).
    """
    # Apply feature engineering
    engineered_input_df = create_engineered_features(raw_input_df.copy())

    # Ensure the final input DataFrame has the exact same columns and order as the
    # X DataFrame used for training the model *before* the ColumnTransformer.
    # This is CRUCIAL for the ColumnTransformer inside the pipeline.
    try:
        final_input_df = engineered_input_df[COLUMNS_ORDER]
    except KeyError as e:
        st.error(f"Column mismatch: A required column for prediction is missing or misnamed. Check COLUMNS_ORDER and feature engineering. Missing: {e}")
        st.stop()

    # Make prediction using the loaded pipeline
    # The pipeline automatically handles scaling and one-hot encoding for the original categorical columns
    prediction_proba = model_pipeline.predict_proba(final_input_df)[:, 1][0]
    prediction = model_pipeline.predict(final_input_df)[0]

    return prediction, prediction_proba

def plot_churn_insights(df: pd.DataFrame):
    """Generates and displays visualizations from the training data related to churn."""
    if df is None:
        st.warning("Training data not available for insights.")
        return

    st.subheader("📊 Insights from Training Data")
    st.markdown("Here are some trends observed from the historical customer data regarding churn:")

    # Plot 1: Churn by Geography
    st.markdown("##### Churn Rate by Geography")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Geography', hue='Exited', palette='viridis', ax=ax1)
    ax1.set_title('Churn Count by Geography')
    ax1.set_xlabel('Geography')
    ax1.set_ylabel('Count')
    ax1.legend(title='Exited', labels=['No Churn', 'Churn'])
    st.pyplot(fig1)
    plt.close(fig1) # Close figure to prevent warning

    # Plot 2: Churn by Gender
    st.markdown("##### Churn Rate by Gender")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='Gender', hue='Exited', palette='magma', ax=ax2)
    ax2.set_title('Churn Count by Gender')
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Count')
    ax2.legend(title='Exited', labels=['No Churn', 'Churn'])
    st.pyplot(fig2)
    plt.close(fig2)

    # Plot 3: Churn by Number of Products
    st.markdown("##### Churn Rate by Number of Products")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='NumOfProducts', hue='Exited', palette='cividis', ax=ax3)
    ax3.set_title('Churn Count by Number of Products')
    ax3.set_xlabel('Number of Products')
    ax3.set_ylabel('Count')
    ax3.legend(title='Exited', labels=['No Churn', 'Churn'])
    st.pyplot(fig3)
    plt.close(fig3)

    # Plot 4: Distribution of Age for Churned vs. Non-Churned
    st.markdown("##### Age Distribution for Churned vs. Non-Churned Customers")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='Age', hue='Exited', fill=True, common_norm=False, palette='viridis', ax=ax4)
    ax4.set_title('Age Distribution by Churn Status')
    ax4.set_xlabel('Age')
    ax4.set_ylabel('Density')
    ax4.legend(title='Exited', labels=['No Churn', 'Churn'])
    st.pyplot(fig4)
    plt.close(fig4)

    # Plot 5: Distribution of Balance for Churned vs. Non-Churned (only if Balance > 0)
    st.markdown("##### Balance Distribution for Churned vs. Non-Churned Customers (Balance > 0)")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df[df['Balance'] > 0], x='Balance', hue='Exited', fill=True, common_norm=False, palette='magma', ax=ax5)
    ax5.set_title('Balance Distribution by Churn Status (for customers with balance)')
    ax5.set_xlabel('Balance (USD)')
    ax5.set_ylabel('Density')
    ax5.legend(title='Exited', labels=['No Churn', 'Churn'])
    st.pyplot(fig5)
    plt.close(fig5)

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

    # Load training data for insights and display plots
    training_df_insights = load_training_data()
    if training_df_insights is not None:
        plot_churn_insights(training_df_insights)
    else:
        st.info("To enable visual insights, please ensure 'train.csv' is in the same directory as your app.")


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

        The **'Insights from Training Data'** section provides visualizations of key trends and distributions from the original dataset, helping you understand the underlying patterns related to customer churn.
        """)
    st.markdown("---")
    st.caption("Developed by Shubham Sharma as a Data Science Portfolio Project.")

if __name__ == '__main__':
    main()
