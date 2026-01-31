import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Basic Page Configuration
st.set_page_config(page_title="AI Expense Detective", layout="wide")

st.title("🔍 AI-Powered Expense Anomaly Detector")
st.markdown("Upload your spending CSV to find 'Wants', 'Needs', and 'Anomalies' automatically.")

# --- SIDEBAR: UPLOAD DATA ---
uploaded_file = st.sidebar.file_uploader("Upload your expense CSV", type=["csv"])

if uploaded_file is not None:
    # 1. Load the Data
    df = pd.read_csv(uploaded_file)
    
    # 2. Preprocessing Logic
    # We convert categories to numbers because AI doesn't read text
    le = LabelEncoder()
    df_proc = df.copy()
    df_proc['Category_Enc'] = le.fit_transform(df['Category'])
    
    # Convert dates to a numerical format the model can process
    df_proc['Date_Ord'] = pd.to_datetime(df['Date']).apply(lambda x: x.toordinal())
    
    # Scale the features so the AI treats 'Amount' and 'Date' fairly
    features = ['Date_Ord', 'Category_Enc', 'Amount']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_proc[features])
    
    # 3. AI Model (Isolation Forest)
    # Contamination=0.05 means we expect roughly 5% of data to be "weird"
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(X_scaled)
    
    # 4. Human-Readable Categorization
    def quick_cat(row):
        if row['Anomaly_Score'] == -1: 
            return 'Waste/Anomaly'
        if row['Category'] in ['Rent', 'Utilities', 'Groceries']: 
            return 'Needs'
        return 'Wants/Essentials'
    
    df['Status'] = df.apply(quick_cat, axis=1)

    # 5. Dashboard Metrics (Top Row)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spending", f"${df['Amount'].sum():,.2f}")
    col2.metric("Anomalies Found", len(df[df['Anomaly_Score'] == -1]))
    col3.metric("Top Category", df['Category'].mode()[0])

    # 6. Visualizations
    st.subheader("Financial Insights")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter Plot: Visualizing Anomalies over Time
    sns.scatterplot(data=df, x='Date', y='Amount', hue='Status', palette='magma', ax=ax[0])
    ax[0].set_title("Spending Patterns & Anomalies")
    plt.setp(ax[0].get_xticklabels(), rotation=45)

    # Pie Chart: Budget Distribution
    budget_dist = df.groupby('Status')['Amount'].sum()
    ax[1].pie(budget_dist, labels=budget_dist.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
    ax[1].set_title("Budget Breakdown")

    st.pyplot(fig)

    # 7. Data Table of Anomalies
    st.subheader("🚩 Detected Anomalies List")
    anomalies_only = df[df['Anomaly_Score'] == -1]
    st.write(anomalies_only)

    # 8. EXPORT AI REPORT (Fixed: Now inside the IF block)
    st.divider()
    st.subheader("📥 Export AI Report")

    @st.cache_data
    def convert_df(df_to_convert):
        # Cache the conversion to make the app faster
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv_data = convert_df(anomalies_only)

    st.download_button(
        label="Download Anomalies as CSV",
        data=csv_data,
        file_name='detected_anomalies_report.csv',
        mime='text/csv',
    )

else:
    # This message shows when the app first loads
    st.info("Waiting for CSV upload. Please use the sidebar to upload your 'synthetic_expenses.csv' file.")