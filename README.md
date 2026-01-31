# AI-Powered Predictive Expense & Anomaly Detector

## 🚀 Project Overview
An end-to-end Data Science pipeline that transforms raw financial data into actionable insights. This project uses Unsupervised Machine Learning to identify fraudulent or unusual spending patterns and provides a business dashboard for budget categorization.

## 🛠️ Tech Stack
- **Language:** Python 3.13
- **Libraries:** Pandas, NumPy, Scikit-Learn (ML), Matplotlib, Seaborn (Visualization)
- **Model:** Isolation Forest (Anomaly Detection)

## 📊 Key Features
- **Synthetic Data Engine:** Generates realistic financial transactions with injected anomalies.
- **Automated Preprocessing:** Handles feature scaling (StandardScaler) and label encoding for categorical data.
- **Anomaly Detection:** Uses Isolation Forest to isolate high-risk transactions.
- **Business Dashboard:** Visualizes spending trends, budget distribution (Needs vs. Wants), and model performance.

## 📂 Structure
- `/data`: Contains raw, processed, and final result CSVs.
- `/src`: Contains the Python source code for generation, processing, and modeling.

## 📈 Results
The model successfully identifies:
1. **Point Anomalies:** Unexpected high-value splurges.
2. **Collective Anomalies:** Duplicate billing events (Identical amounts/categories on same days).