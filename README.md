# 🔍 Zia AI: Predictive Expense & Anomaly Detector

**Live App Link:** [Click here to view the App](https://predictive-expense-analyzer-zia-ai.streamlit.app/)

## 🚀 Executive Summary
Zia AI is an intelligent financial security tool designed to detect "Financial Leakage." While standard banking apps just track spending, Zia AI uses **Unsupervised Machine Learning** to proactively identify unusual patterns—such as forgotten subscriptions, billing errors, or impulsive splurges—that could be costing users thousands annually.

## 📊 Key Features
- **Intelligent Anomaly Hunting:** Uses the **Isolation Forest** algorithm to detect fraudulent or unusual transactions.
- **Budget Categorization:** Automatically classifies spending into **Needs**, **Wants**, and **Waste**.
- **Data Engineering Pipeline:** Includes automated feature scaling (`StandardScaler`) and label encoding.
- **Interactive Dashboard:** Real-time visualization of spending trends and budget distribution.



## 🧠 The Tech Stack & Logic
- **Language:** Python 3.13
- **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud
- **AI Model:** **Isolation Forest**. I chose this model because it is highly efficient at isolating "outliers" in multi-dimensional datasets without requiring manual labeling.



## 📂 Project Structure
- `/src/app.py`: The core Streamlit application and ML logic.
- `requirements.txt`: Project dependencies for cloud deployment.
- `synthetic_expenses.csv`: Example dataset used for model validation.

## 📈 Business Results
The model successfully identifies:
1. **Global Anomalies:** Unexpected high-value splurges.
2. **Contextual Anomalies:** Spending that doesn't fit the user's historical category patterns.
3. **Budget Leakage:** Flags "Waste" items that allow users to reclaim ~10% of their monthly income.

## 📥 Installation & Local Setup
1. Clone the repository: `git clone <your-repo-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the app: `streamlit run src/app.py`