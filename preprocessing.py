# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables

def preprocess_data(df, train_mode=True):
    """
    Preprocesses a DataFrame for machine learning tasks.
    
    Parameters:
    df (DataFrame): Input data to be processed
    train_mode (bool): Whether this is training data (has target variable)
    
    Returns:
    DataFrame: Processed features
    Series: Target variable (if train_mode=True and 'churned' exists)
    """
    
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # STEP 1: Handle target variable upfront
    target_col = 'churned'
    y = None
    
    if target_col in df.columns:
        if train_mode:
            y = df[target_col]  # Store target for training
        df = df.drop(columns=[target_col])  # Always remove from features
    
    # STEP 2: Basic cleaning
    df.drop_duplicates(inplace=True)
    
    # Remove ID columns and low-variance features
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df.drop(columns=id_cols, inplace=True, errors='ignore')
    df = df.loc[:, df.nunique() > 1]
    
    # STEP 3: Handle specific columns
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # STEP 4: Missing value imputation
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # STEP 5: Custom encoding for service features
    service_mapping = {
        'No internet service': 0,
        'No phone service': 0,
        'No': 1,
        'Yes': 2,
        False: 1,
        True: 2
    }
    
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    
    for col in service_cols:
        if col in df.columns:
            df[f'{col}_enc'] = df[col].map(service_mapping)
    
    # STEP 6: Binary columns conversion
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0, True: 1, False: 0})
            df[col] = df[col].infer_objects(copy=False)
    
    # STEP 7: Gender encoding
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0}).infer_objects(copy=False)
    
    # STEP 8: One-hot encoding for multi-class features
    multi_class_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, 
                       columns=[col for col in multi_class_cols if col in df.columns], 
                       drop_first=True)
    
    # STEP 9: Handle remaining categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
        else:
            df[col] = df[col].astype('category').cat.codes
    
    # STEP 10: Convert booleans to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    # Return appropriate values based on mode
    return (df, y) if train_mode and y is not None else df