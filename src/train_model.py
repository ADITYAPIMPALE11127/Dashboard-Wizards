"""
Telco Customer Churn Prediction - Optimized Version with Error Fixes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import resample
from sklearn.metrics import (classification_report, roc_auc_score,
                           confusion_matrix, precision_recall_curve,
                           average_precision_score,balanced_accuracy_score)
from preprocessing import preprocess_data
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform

# Visualization setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluates model performance with customizable threshold"""
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    print("\n📊 Classification Report (threshold={:.2f}):".format(threshold))
    print(classification_report(y_test, y_pred))
    
    print(f"📈 AUC-ROC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"🎯 Average Precision Score: {average_precision_score(y_test, y_pred_prob):.4f}")
    # Add balanced accuracy metric:
    print(f"⚖️ Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
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

def find_optimal_threshold(y_true, y_pred_prob):
    """Determines optimal classification threshold"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure()
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1')
    plt.axvline(optimal_threshold, color='k', linestyle='--')
    plt.title('Precision-Recall Tradeoff')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    
    return optimal_threshold

def train_with_hyperparameters(X_train, y_train):
    """Performs hyperparameter tuning using RandomizedSearchCV"""
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None] + list(np.arange(10, 100, 10)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True],  # Changed to only True to avoid conflict with max_samples
        'class_weight': [{0:1, 1:2}, {0:1, 1:3}, 'balanced', None],
        'max_samples': uniform(0.5, 0.5)
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='roc_auc',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("🔍 Starting hyperparameter tuning...")
    search.fit(X_train, y_train)
    
    print("\n✅ Best parameters found:")
    for param, value in search.best_params_.items():
        print(f"{param}: {value}")
    
    return search.best_estimator_

def train_in_chunks(data_path, chunk_size=2000, test_size=0.2, tune_hyperparams=False):
    """Trains model on balanced chunks with optional hyperparameter tuning"""
    # First pass to get test set
    print("🔍 Preparing test set...")
    full_data = pd.read_csv(data_path)
    X, y = preprocess_data(full_data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=test_size,
        random_state=42
    )
    
    # Hyperparameter tuning on first chunk
    if tune_hyperparams:
        print("🧪 Tuning hyperparameters on first chunk...")
        first_chunk = pd.read_csv(data_path, nrows=chunk_size)
        X_chunk, y_chunk = preprocess_data(first_chunk)
        model = train_with_hyperparameters(X_chunk, y_chunk)
    else:
        model = RandomForestClassifier(
            class_weight={0: 1, 1: 3},
            random_state=42,
            n_jobs=-1,
            warm_start=True
        )
    
    # Process training data in chunks
    print("🔄 Processing training chunks...")
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size)):
        # Skip rows that are in test set (fixed condition)
        test_indices = set(X_test.index)
        chunk_indices = set(chunk.index)
        if chunk_indices.intersection(test_indices):
            continue
            
        X_chunk, y_chunk = preprocess_data(chunk)
        df_chunk = pd.concat([X_chunk, y_chunk], axis=1)
        majority = df_chunk[df_chunk['churned'] == 0]
        minority = df_chunk[df_chunk['churned'] == 1]
        
        majority_downsampled = resample(majority,
                                      replace=False,
                                      n_samples=int(len(minority)*1.5),
                                      random_state=42)
        
        balanced_chunk = pd.concat([majority_downsampled, minority])
        X_balanced = balanced_chunk.drop('churned', axis=1)
        y_balanced = balanced_chunk['churned']
        
        if not hasattr(model, 'estimators_'):
            model.n_estimators = 100
            model.fit(X_balanced, y_balanced)
        else:
            model.n_estimators += 50
            model.fit(X_balanced, y_balanced)
        
        print(f"✅ Processed chunk {i+1} | Current trees: {model.n_estimators}")
    
    return model, X_test, y_test

def main():
    """Main program execution"""
    DATA_PATH = '../data/telco_train.csv'
    
    # Set to True for hyperparameter tuning (better accuracy but slower)
    TUNE_HYPERPARAMS = True
    
    print("🏗️ Building model...")
    model, X_test, y_test = train_in_chunks(
        DATA_PATH,
        chunk_size=2000,
        tune_hyperparams=TUNE_HYPERPARAMS
    )
    
    print("\n🧪 Evaluating model...")
    y_pred_prob = evaluate_model(model, X_test, y_test)
    
    print("\n🔎 Analyzing important features...")
    plot_feature_importance(model, X_test.columns)
    
    print("\n⚖️ Optimizing decision threshold...")
    optimal_threshold = find_optimal_threshold(y_test, y_pred_prob)
    print(f"\n🔧 Optimal threshold: {optimal_threshold:.4f}")
    
    print("\n🏆 Final Evaluation:")
    evaluate_model(model, X_test, y_test, threshold=optimal_threshold)
    
    model_name = "churn_model_tuned.pkl" if TUNE_HYPERPARAMS else "churn_model.pkl"
    joblib.dump(model, model_name)
    print(f"\n💾 Saved model as '{model_name}'")

if __name__ == "__main__":
    main()