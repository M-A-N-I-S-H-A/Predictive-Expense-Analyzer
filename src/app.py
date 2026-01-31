import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Expense Detective", layout="wide")

st.title("🔍 AI-Powered Expense Anomaly Detector")
st.markdown("Upload your spending CSV to find 'Wants', 'Needs', and 'Anomalies' automatically.")

# 1. Sidebar - Upload Data
uploaded_file = st.sidebar.file_uploader("Upload your expense CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 2. Preprocessing Logic
    le = LabelEncoder()
    df_proc = df.copy()
    df_proc['Category_Enc'] = le.fit_transform(df['Category'])
    df_proc['Date_Ord'] = pd.to_datetime(df['Date']).apply(lambda x: x.toordinal())
    
    features = ['Date_Ord', 'Category_Enc', 'Amount']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_proc[features])
    
    # 3. AI Model (Isolation Forest)
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(X_scaled)
    
    # Categorize
    def quick_cat(row):
        if row['Anomaly_Score'] == -1: return 'Waste/Anomaly'
        if row['Category'] in ['Rent', 'Utilities']: return 'Needs'
        return 'Wants/Essentials'
    
    df['Status'] = df.apply(quick_cat, axis=1)

    # 4. Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spending", f"${df['Amount'].sum():,.2f}")
    col2.metric("Anomalies Found", len(df[df['Anomaly_Score'] == -1]))
    col3.metric("Top Expense", df['Category'].mode()[0])

    # 5. Visualizations
    st.subheader("Financial Insights")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter Plot
    sns.scatterplot(data=df, x='Date', y='Amount', hue='Status', palette='magma', ax=ax[0])
    ax[0].set_title("Spending Over Time")
    plt.xticks(rotation=45)

    # Pie Chart
    budget_dist = df.groupby('Status')['Amount'].sum()
    ax[1].pie(budget_dist, labels=budget_dist.index, autopct='%1.1f%%', startangle=140)
    ax[1].set_title("Budget Breakdown")

    st.pyplot(fig)

    # 6. Show Data Table
    st.subheader("Detected Anomalies")
    st.write(df[df['Anomaly_Score'] == -1])
else:
    st.info("Waiting for CSV upload. You can use the 'synthetic_expenses.csv' from your data folder!")
    
# 7. Add Download Button for Anomalies
st.divider()
st.subheader("📥 Export AI Report")

# Convert the anomaly dataframe to CSV
anomalies_only = df[df['Anomaly_Score'] == -1]

@st.cache_data
def convert_df(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

csv_data = convert_df(anomalies_only)

st.download_button(
    label="Download Anomalies as CSV",
    data=csv_data,
    file_name='detected_anomalies.csv',
    mime='text/csv',
)