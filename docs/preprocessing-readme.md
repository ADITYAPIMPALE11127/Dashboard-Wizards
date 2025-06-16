
```markdown
# Data Preprocessing Documentation

## preprocess_data(df, train_mode=True)

# Data Preprocessing Documentation

## Function: `preprocess_data(df, train_mode=True)`

### Parameters

| Parameter   | Type            | Description                                                                 |
|-------------|-----------------|-----------------------------------------------------------------------------|
| `df`        | pandas.DataFrame | Input raw data (expected to contain Telco customer churn features)          |
| `train_mode`| bool            | If `True` (default), handles target variable `'churned'` separation        |

### Returns

| Return      | Type            | Description                                                                 |
|-------------|-----------------|-----------------------------------------------------------------------------|
| `df`        | pandas.DataFrame | Processed feature matrix with all numeric values                            |
| `y`         | pandas.Series    | Target variable series (only returned when `train_mode=True`)               |

## Processing Pipeline

1. **Initial Setup**  
   - Creates copy of input DataFrame
   - Handles target variable separation if present and `train_mode=True`

2. **Data Cleaning**  
   ```python
   # Remove duplicates and ID columns
   df.drop_duplicates(inplace=True)
   id_cols = [col for col in df.columns if 'id' in col.lower()]
   df.drop(columns=id_cols, inplace=True, errors='ignore')
   
   # Remove constant columns
   df = df.loc[:, df.nunique() > 1]

### 3. Column-Specific Processing
**TotalCharges:**
```python
# Convert to numeric + smart impute
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)
```

### 4. Missing Value Treatment
| Data Type | Method | Example Columns |
|-----------|--------|-----------------|
| Numeric | Median | tenure, MonthlyCharges |
| Categorical | Mode | PaymentMethod, Contract |

### 5. Feature Encoding
**Service Features Encoding:**
```python
service_mapping = {
    'No internet service': 0,
    'No phone service': 0,
    'No': 1,
    'Yes': 2
}
# Applied to: OnlineSecurity, OnlineBackup, DeviceProtection, etc.
```

**Other Encodings:**
| Original | Encoded | Columns |
|----------|---------|---------|
| Yes/No | 1/0 | Partner, Dependents |
| Male/Female | 1/0 | gender |
| Multi-class | One-hot | InternetService, Contract |

---

## Output Specifications
### Generated Columns
- All original columns (cleaned)
- `*_enc` columns for service features
- One-hot encoded categoricals (prefixes: `Contract_`, `PaymentMethod_`, etc.)
- All values converted to numeric

### Expected Output Shape
- Features: (n_samples, n_features) 
  - Typical: ~20 columns after processing
- Target: (n_samples,) [when train_mode=True]

---

## Dependencies
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
```

## Usage Example
```python
# For training
X_train, y_train = preprocess_data(raw_train_df, train_mode=True)

# For inference
X_test = preprocess_data(raw_test_df, train_mode=False)
```

## Notes
1. Test data must contain all columns present in training data
2. No feature scaling is applied by default
3. Boolean columns automatically converted to int (1/0)
```
