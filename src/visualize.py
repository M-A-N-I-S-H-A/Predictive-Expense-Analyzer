import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the results
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'final_results.csv')
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])

# 2. Set the style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# 3. Create a Scatter Plot of spending
# Normal points in Blue, Anomalies in Red
sns.scatterplot(data=df, x='Date', y='Amount', hue='Anomaly_Score', 
                palette={1: 'blue', -1: 'red'}, alpha=0.6)

plt.title('AI-Detected Spending Anomalies', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.legend(title='Detection', labels=['Normal', 'Anomaly'])

# 4. Save the plot
plot_path = os.path.join(current_dir, '..', 'data', 'spending_analysis.png')
plt.savefig(plot_path)
print(f"Chart saved successfully at: {plot_path}")

# 5. Show the plot
plt.show()