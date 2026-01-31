import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load Data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'final_results.csv')
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])

# 2. Categorize into Needs, Wants, Waste
def categorize_budget(row):
    if row['Anomaly_Score'] == -1: return 'Waste/Anomaly'
    if row['Category'] in ['Rent', 'Utilities']: return 'Needs'
    if row['Category'] in ['Entertainment', 'Shopping']: return 'Wants'
    return 'Essentials (Food/Transport)'

df['Budget_Type'] = df.apply(categorize_budget, axis=1)

# 3. Create the Dashboard (3 Charts)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.4)

# CHART 1: Pie Chart (Distribution of Spending)
budget_counts = df.groupby('Budget_Type')['Amount'].sum()
axes[0, 0].pie(budget_counts, labels=budget_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
axes[0, 0].set_title('Spending Distribution (Needs vs Wants)')

# CHART 2: Line Graph (Spending Over Time)
daily_spend = df.groupby('Date')['Amount'].sum().reset_index()
sns.lineplot(data=daily_spend, x='Date', y='Amount', ax=axes[0, 1], color='blue')
axes[0, 1].set_title('Daily Spending Trend')
axes[0, 1].tick_params(axis='x', rotation=45)

# CHART 3: Bar Chart (Predicted vs Actual)
# For this simulation, we'll assume "Predicted" was a flat budget of $70/day
actual_avg = df['Amount'].mean()
predicted_avg = 65.0  # Simulated prediction
axes[1, 0].bar(['Actual Avg', 'Predicted Avg'], [actual_avg, predicted_avg], color=['blue', 'green'])
axes[1, 0].set_title('Model Performance: Actual vs Predicted Avg')

# Remove the empty 4th subplot
fig.delaxes(axes[1, 1])

# 4. Save and Show
plt.savefig(os.path.join(current_dir, '..', 'data', 'final_dashboard.png'))
print("Dashboard saved to data/final_dashboard.png")
plt.show()