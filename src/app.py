import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Basic Page Configuration
st.set_page_config(page_title="AI Expense Detective Pro", layout="wide")

# Custom CSS to make metrics look like 'Cards'
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 AI-Powered Expense Anomaly Detector")
st.markdown("Upload your spending CSV to find 'Wants', 'Needs', and 'Anomalies' automatically.")

# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("Setup")
uploaded_file = st.sidebar.file_uploader("Upload your expense CSV", type=["csv"])

if uploaded_file is not None:
    # 2. Load the Data
    df = pd.read_csv(uploaded_file)
    
    # 3. Preprocessing Logic
    le = LabelEncoder()
    df_proc = df.copy()
    df_proc['Category_Enc'] = le.fit_transform(df['Category'])
    df_proc['Date_Ord'] = pd.to_datetime(df['Date']).apply(lambda x: x.toordinal())
    
    # Scaling Features
    features = ['Date_Ord', 'Category_Enc', 'Amount']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_proc[features])
    
    # 4. AI Model (Isolation Forest)
    model = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(X_scaled)
    
    # Human-Readable Categorization
    def quick_cat(row):
        if row['Anomaly_Score'] == -1: 
            return 'Waste/Anomaly'
        if row['Category'] in ['Rent', 'Utilities', 'Groceries']: 
            return 'Needs'
        return 'Wants/Essentials'
    
    df['Status'] = df.apply(quick_cat, axis=1)

    # 5. Dashboard Metrics (Cards at the top)
    # Using a container with border for the metrics row
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Spending", f"${df['Amount'].sum():,.2f}")
        col2.metric("Anomalies Found", len(df[df['Anomaly_Score'] == -1]))
        col3.metric("Top Category", df['Category'].mode()[0])

    st.divider()

    # 6. Visualizations (Organized into Two Columns with Cards)
    st.subheader("📊 Financial Intelligence Dashboard")
    
    col_left, col_right = st.columns(2)

    # Custom Professional Color Palette
    custom_colors = {"Needs": "#00d4ff", "Wants/Essentials": "#9b59b6", "Waste/Anomaly": "#e74c3c"}

    with col_left:
        with st.container(border=True):
            st.markdown("### 📈 Spending Patterns")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            # Set dark background for the plot to match our theme
            fig1.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#0e1117')
            
            sns.scatterplot(data=df, x='Date', y='Amount', hue='Status', palette=custom_colors, s=120, ax=ax1)
            
            # Formatting plot text color for dark mode
            ax1.tick_params(colors='white')
            ax1.xaxis.label.set_color('white')
            ax1.yaxis.label.set_color('white')
            plt.setp(ax1.get_xticklabels(), rotation=45)
            
            st.pyplot(fig1)

    with col_right:
        with st.container(border=True):
            st.markdown("### 🍕 Budget Allocation")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            fig2.patch.set_facecolor('#0e1117')
            
            budget_dist = df.groupby('Status')['Amount'].sum()
            # Modern Donut Chart
            ax2.pie(budget_dist, labels=budget_dist.index, autopct='%1.1f%%', 
                    startangle=140, colors=["#e74c3c", "#00d4ff", "#9b59b6"],
                    wedgeprops={'width': 0.4}, textprops={'color':"w"}) 
            
            st.pyplot(fig2)

    # 7. Data Table of Anomalies (Inside its own Card)
    with st.container(border=True):
        st.subheader("🚩 Detected Anomalies List")
        anomalies_only = df[df['Anomaly_Score'] == -1]
        st.dataframe(anomalies_only, use_container_width=True)

    # 8. EXPORT AI REPORT
    st.divider()
    st.subheader("📥 Export AI Report")

    @st.cache_data
    def convert_df(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv_data = convert_df(anomalies_only)

    st.download_button(
        label="Download Anomalies as CSV",
        data=csv_data,
        file_name='detected_anomalies_report.csv',
        mime='text/csv',
        type="primary" # Makes the button blue/highlighted
    )

else:
    # Landing Page State
    st.info("👋 Welcome! Please upload your 'synthetic_expenses.csv' file in the sidebar to begin the analysis.")
    # Show a placeholder image or tip
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1000", caption="AI Analysis will appear here")