# 🛑 Stop the Churn – Fintech User Churn Prediction

A machine learning web application that predicts which fintech users are likely to **churn** (stop using the app), and displays those predictions along with key insights in an interactive **Streamlit dashboard**.

## 🔗 Quick Links  
- [Data Preprocessing](docs/preprocessing-readme.md)  
- [Project PDF Explanation](https://drive.google.com/...)  
- [Canva Presentation](https://www.canva.com/...)  
---

## 📈 Project Overview

The Streamlit dashboard provides:

- **Bar charts** for visualizing the distribution of churn-risk scores across users.
- **Pie charts** to show proportions of low-, medium-, and high-risk users.
- A **ranked top-10 high-risk user table** for targeted retention efforts.
- A feature to **download the high-risk user list** in CSV or Excel format for easy sharing and follow-up.

These interactive visuals and data export options transform raw predictions into intuitive, actionable insights—helping teams focus retention strategies on the most at-risk users.

---

## 🔍 Problem Statement

Customer churn is a critical challenge in fintech. Retaining users is more cost-effective than acquiring new ones. This project aims to:

- Predict churn risk for each user using machine learning
- Provide meaningful insights for product and marketing teams
- Help improve retention strategies

---

## 📊 Project Features

✅ Predicts churn probability for each user  
✅ Displays a ranked list of high‑risk users  
✅ Shows data‑driven insights (charts, tables)  
✅ Built with an intuitive Streamlit web dashboard  

---

## 📁 Project Structure

Dashboard-Wizards/
├── app.py                    # Main Streamlit dashboard
├── train_model.py           # Model training script
├── churn_model_best.pkl     # Trained ML model
├── preprocessing.py         # Preprocessing pipeline (Telco & fintech)
├── preprocessing-readme.md  # Documentation for preprocessing steps
├── data/                    # Raw & sample datasets
├── test_debug/              # Debugging/test utilities
└── __pycache__/             # Python cache (can be ignored)

---

## ⚙️ Tech Stack

| Purpose       | Technology                   | Version         |
|---------------|------------------------------|-----------------|
| ML Model      | Scikit‑learn - 1.6.1  (Random Forest) | 1.6.1  |
| Data Handling | Pandas, NumPy                | 2.0.2           |
| Visualizations| Matplotlib, Seaborn          | 3.9.4 , 0.13.2  |
| Web Interface | Streamlit                    | 1.45.1          |
| Libraries     | SciPy, Joblib                | 1.13.2 , 1.5.1  |

---

## 🔧 Installation & Setup

Clone the repository and set up your environment:

git clone https://github.com/ADITYAPIMPALE11127/Dashboard-Wizards.git
cd Dashboard-Wizards

---

## 🚀 How to Run the Dashboard

streamlit run app.py
Browse to http://localhost:8501 to view the dashboard and interact with the visualizations and data download options.

---

## 🧠 Model Training
To retrain or update the churn model:

python train_model.py
This script will:

- Load and preprocess data using preprocessing.py
- Train a Random Forest model
- Save the model to churn_model_best.pkl for use in app.py

---

## 🔧 Telco Data Preprocessing Module
This module cleans and prepares Telco Customer Churn data for ML:

from preprocessing import load_and_clean_data

df = load_and_clean_data("data/telco.csv", save_cleaned_csv=True)
Key steps include:

- Removing duplicates and ID columns
- Converting numeric fields (TotalCharges) and filling missing values
- Encoding categorical features (binary Yes/No, gender, service-level, and one-hot encoding)
- Saving a cleansed version as cleaned_data.csv
  

## 📩 Contact
Created by Adithya Pimpale and Chaithra shree
📧 pimpaleaditya2@gmail.com


