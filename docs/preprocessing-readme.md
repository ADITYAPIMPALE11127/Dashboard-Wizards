# Telco Customer Churn - Data Preprocessing

This project focuses on cleaning and preparing the Telco Customer Churn dataset for building machine learning models. The code is modular, readable, and designed to help teams easily plug into the cleaned dataset for analysis and training.

---

## 📁 Files Overview

| File Name            | Description |
|----------------------|-------------|
| `preprocessing.py`   | Main script to load, clean, and preprocess the dataset. |
| `cleaned_data.csv`   | (Optional) Output file generated after preprocessing the input dataset. |
| `main-test.ipynb | to run the preprocessing.py file and for testing and debuging |
---

## ▶️ How to Use

1. Place your raw dataset CSV file (e.g., `telco.csv`) in the same folder as `preprocessing.py`.

2. Run the following Python code:

```python
from preprocessing import load_and_clean_data

# Load and clean data
df = load_and_clean_data("telco.csv", save_cleaned_csv=True)
This will:

Return a cleaned and preprocessed DataFrame df

Save a cleaned version as cleaned_data.csv (if save_cleaned_csv=True)

🔍 What load_and_clean_data() Does
Step	Description
1️⃣	Load the CSV file into a pandas DataFrame
2️⃣	Remove duplicate records
3️⃣	Drop columns that look like IDs (columns with 'id' in name)
4️⃣	Remove constant columns (same value for all rows)
5️⃣	Convert TotalCharges to numeric
6️⃣	Fill missing values: median for numeric columns, mode for categorical columns
7️⃣	Create new encoded columns for special services like StreamingTV, OnlineSecurity, etc.
8️⃣	Convert Yes/No columns (Partner, PhoneService, etc.) to 1/0
9️⃣	Encode gender column: Male → 1, Female → 0
🔟	One-hot encode categorical columns like InternetService, Contract, PaymentMethod
🔁	Label encode any remaining binary string columns
🔚	Convert boolean-type columns to integers

🔢 Special Encoding Explanation
For service-related columns like:

OnlineSecurity

OnlineBackup

DeviceProtection

TechSupport

StreamingTV

StreamingMovies

MultipleLines

We created new columns (e.g., OnlineSecurity_enc) using the mapping below:

Original Value	Encoded As	Meaning
No internet service	0	Service not applicable (no internet)
No phone service	0	Service not applicable (no phone line)
No	1	Service available but not used
Yes	2	Service available and in use
False (boolean)	1	Treated like "No"
True (boolean)	2	Treated like "Yes"

This helps the model distinguish between:

🚫 Not applicable service (0)

❌ Declined but available service (1)

✅ Accepted and in use service (2)

✅ Additional Encoding Summary
Binary Yes/No columns → replaced with 1 (Yes) / 0 (No)

Gender: Male → 1, Female → 0

Multi-class text columns (InternetService, Contract, PaymentMethod) → one-hot encoded

Remaining binary object columns → label encoded if they have exactly 2 unique values

Boolean columns → converted to integers (1 for True, 0 for False)

🧠 For Model Training
Make sure to encode the target column Churn as Yes → 1 and No → 0 (if not already handled).

You can use the cleaned data from the returned df or directly from cleaned_data.csv.

Optionally, scale numeric features like tenure, MonthlyCharges, and TotalCharges before model training.

