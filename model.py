# model.py
import os
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_model(model_path='churn_model_tuned.pkl'):
    """Load the trained model with error handling"""
    try:
        # Verify the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {os.path.abspath(model_path)}\n"
                "Please ensure you've trained the model first or placed the "
                ".pkl file in the correct location."
            )
        
        # Load and return the model
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluates model performance with customizable threshold"""
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    print("\n📊 Classification Report (threshold={:.2f}):".format(threshold))
    print(classification_report(y_test, y_pred))
    
    print(f"📈 AUC-ROC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    return y_pred_prob


def plot_feature_importance(model, feature_names, top_n=20):
    """Visualizes the most important features"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure()
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()