# 🛑 Stop the Churn – Fintech User Churn Prediction

A machine learning web application that predicts which fintech users are likely to **churn** (stop using the app), and displays those predictions along with key insights in an interactive **Streamlit dashboard**.

## 🔗 Quick Links  
- [Data Preprocessing](docs/preprocessing-readme.md)  
- [Project Presentation Video Link](https://drive.google.com/drive/folders/13kknPg4igC-iTFqLWqi1QMuYN8KFMJtV?usp=sharing)  
- [Canva Presentation File PPT](https://www.canva.com/design/DAGqUJlr3WQ/KOPINYo16O4w-OzpDugrcg/edit?utm_content=DAGqUJlr3WQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)  
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

```bash
DASHBOARD-WIZARD/
│
├── 📁 __pycache__/                  # Python cache files
│
├── 📁 data/                         # Directory for storing raw or processed datasets
│
├── 📁 docs/                         # Project documentation
│   └── preprocessing-readme.md     # Notes on preprocessing steps
│
├── 📁 eda/                          # Notebooks for Exploratory Data Analysis
│   ├── main_test.ipynb             # EDA notebook for testing main ideas
│   └── visualize.ipynb             # Notebook for visualizing insights
│
├── 📁 models/                       # Trained model storage
│   └── churn_model_tuned.pkl       # Serialized tuned model (pkl file)
│
├── 📁 src/                          # Source code
│   ├── __pycache__/                # Python cache files
│   ├── app.py                      # Streamlit web application entry point
│   ├── churn_model_tuned.pkl       # Local copy of the trained model
│   ├── model.py                    # Model loading and prediction functions
│   ├── preprocessing.py            # Data preprocessing logic
│   └── train_model.py              # Training script for churn model
│
├── 📁 visualized_outputs/          # Output visualizations and charts
│
├── README.md                       # Project overview and usage instructions
├── render.yaml                     # Render deployment configuration
└── requirements.txt                # Python dependencies list

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


