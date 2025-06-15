# app.py (final production-ready version)
import os
os.environ['MPLBACKEND'] = 'Agg'  # Critical for server environments

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Must come before pyplot
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn metrics
from sklearn.metrics import (
    roc_curve,
    auc,
)

# Local application imports
from preprocessing import preprocess_data
from model import (
    load_model,
    evaluate_model,
    plot_feature_importance
)
# Page config
st.set_page_config(
    page_title="Churn Prediction Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHURN_THRESHOLD = 0.6  # Default classification threshold
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/churn_model_tuned.pkl')
@st.cache_resource
def cached_load_model():
    """Cached model loader with validation"""
    try:
        # Verify model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {os.path.abspath(MODEL_PATH)}")
        
        model = load_model(MODEL_PATH)
        
        # Validate model structure
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Invalid model - missing predict_proba method")
            
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.error("Please ensure you have the trained model file")
        st.stop()

def clean_data(df):
    """Enhanced data cleaning with validation"""
    REQUIRED_COLS = {'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges'}
    
    # Validate input
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Type conversion
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df
# UI PART
def main():
    """Main application flow"""
    st.title("🚀 Customer Churn Prediction Dashboard")
    st.markdown("Upload customer data to identify churn risks")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", 
                        ["Home", "Model Performance", "Customer Analysis"],
                        label_visibility="collapsed")
        
        # Debug info (hidden by default)
        if st.checkbox("Show debug info", False):
            st.write(f"Model path: {os.path.abspath(MODEL_PATH)}")
            st.write(f"Model exists: {os.path.exists(MODEL_PATH)}")

    # File processing page
    if page == "Home":
        st.header("📤 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload customer data (CSV)", 
            type=["csv"],
            help="Should include customer behavior features"
        )
        
        if uploaded_file:
            with st.spinner('Processing data...'):
                try:
                    # Load and validate data
                    df = pd.read_csv(uploaded_file)
                    df = clean_data(df)
                    
                    # Store whether churned column exists (for evaluation)
                    has_churned = 'churned' in df.columns
                    
                    # Preprocess features - EXCLUDE churned column if present
                    X = preprocess_data(df, train_mode=False)
                    
                    # Generate predictions
                    model = cached_load_model()
                    df['churn_probability'] = model.predict_proba(X)[:, 1]
                    
                    # Store in session state
                    st.session_state.update({
                        'processed_data': df,
                        'features': X,
                        'has_churned': has_churned,
                        'original_cols': df.columns.tolist(),
                        'y_true': df['churned'].copy() if has_churned else None
                    })
                    
                    # Show success message
                    st.success("✅ Data processed successfully")
                    if has_churned:
                        st.warning("⚠️ 'churned' column was detected and excluded from predictions")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    st.stop()

    # Model evaluation page
    elif page == "Model Performance" and 'processed_data' in st.session_state:
        st.header("📈 Model Performance")
        df = st.session_state['processed_data']
        X = st.session_state['features']
    
        if st.session_state['has_churned']:
            y_true = st.session_state['y_true']
            y_prob = df['churn_probability']
            
            # Metrics evaluation
            with st.expander("📊 Classification Report", expanded=True):
                model = cached_load_model()
                evaluate_model(model, X, y_true)
            
            # Visualizations
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("📈 ROC Curve"):
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, 
                            label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc="lower right")
                    st.pyplot(plt)
            
            with col2:
                with st.expander("🔍 Feature Importance"):
                    try:
                        plot_feature_importance(model, X.columns)
                    except Exception as e:
                        st.warning(f"Could not plot features: {str(e)}")
        else:
            st.warning("⚠️ Model evaluation requires ground truth data")
            st.info("""
            To see full model performance metrics, please upload data containing:
            - A column named **'churned'** 
            - With values **1** (churned) and **0** (retained)
            """)
            
            # Show sample predictions instead
            with st.expander("🎯 Prediction Distribution", expanded=True):
                fig = plt.figure(figsize=(10, 4))
                sns.histplot(df['churn_probability'], bins=20, kde=True)
                plt.title('Predicted Churn Probabilities Distribution')
                plt.xlabel('Churn Probability')
                st.pyplot(fig)
            
            with st.expander("📊 Top Features", expanded=True):
                try:
                    model = cached_load_model()
                    plot_feature_importance(model, X.columns)
                except Exception as e:
                    st.warning(f"Could not show features: {str(e)}")

    # Customer analysis page
    elif page == "Customer Analysis" and 'processed_data' in st.session_state:
        st.header("🔍 Customer Analysis")

        df = st.session_state['processed_data']
        threshold = st.slider("Churn threshold", 0.0, 1.0, 
                            CHURN_THRESHOLD, 0.01,
                            help="Probability cutoff for churn classification")

        # Filter for unchurned customers only (churned=0 or churned column doesn't exist)
        if 'churned' in df.columns:
            analysis_df = df[df['churned'] == 0].copy()
        else:
            analysis_df = df.copy()

        # Visualization row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Probability Distribution**")
            fig = plt.figure(figsize=(8, 4))
            sns.histplot(analysis_df['churn_probability'], bins=30, kde=True)
            plt.axvline(threshold, color='r', linestyle='--')
            plt.xlabel("Churn Probability")
            st.pyplot(fig)

        with col2:
            st.markdown("**Retention Outlook**")
            fig = plt.figure(figsize=(6, 6))
            churn_counts = analysis_df['churn_probability'].ge(threshold).value_counts()
            plt.pie(churn_counts, 
                    labels=['Retain', 'At Risk'],
                    autopct='%1.1f%%',
                    colors=['#4CAF50', '#FFC107'])
            st.pyplot(fig)

        # Show only top 10 high-risk UNCHURNED customers
        st.markdown("**⚠️ Top 10 At-Risk Customers (Still Active)**")
        high_risk_customers = analysis_df.sort_values(by='churn_probability', ascending=False).head(10).copy()

        cols_to_show = ['churn_probability']
        if 'churned' in st.session_state['original_cols']:
            cols_to_show.append('churned')
        cols_to_show += [c for c in st.session_state['original_cols']
                        if c not in ['churn_probability', 'churned']]
        
        if not high_risk_customers.empty:
            st.dataframe(
                high_risk_customers[cols_to_show]
                .reset_index(drop=True)
                .style.format({'churn_probability': '{:.1%}'})
                .apply(lambda x: ['background: #FFF3CD' if x.name == 'churn_probability' and v >= threshold else '' for v in x], axis=0),
                use_container_width=True
            )
        else:
            st.info("No at-risk customers found above the selected threshold.")
        
        # Full dataset download
        csv_data = df.to_csv(index=False)
        st.download_button(
          label="✨ Download Full Predictions (CSV)",
        data=csv_data,
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
        help="Contains all customer data with churn probabilities",
        use_container_width=True,  # Makes button full width
        type="primary",  # Makes button more prominent (Streamlit >= 1.16)
        key="full_export_btn"  # Unique key to avoid conflicts
        )
        st.caption("File includes: Customer ID, Churn Probability, and all original features")
    elif page in ["Model Performance", "Customer Analysis"]:
        st.warning("Please upload data on the Home page first")

if __name__ == "__main__":
    main()