import pandas as pd
import os
from sklearn.ensemble import IsolationForest

# 1. Load the processed data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'processed_data.csv')
original_path = os.path.join(current_dir, '..', 'data', 'synthetic_expenses.csv')

X = pd.read_csv(data_path)
df_original = pd.read_csv(original_path)

# 2. Initialize the AI Model
# contamination=0.05 means we expect about 5% of data to be anomalies
model = IsolationForest(contamination=0.05, random_state=42)

# 3. Train the model and predict
# The model will return -1 for anomalies and 1 for normal data
df_original['Anomaly_Score'] = model.fit_predict(X)

# 4. Filter the results
anomalies = df_original[df_original['Anomaly_Score'] == -1]

print("--- ANOMALY DETECTION REPORT ---")
print(f"Total Transactions Analyzed: {len(df_original)}")
print(f"Total Anomalies Found: {len(anomalies)}")
print("\nTop 5 Detected Anomalies:")
print(anomalies[['Date', 'Category', 'Amount', 'Label']].head())

# 5. Save the results
output_path = os.path.join(current_dir, '..', 'data', 'final_results.csv')
df_original.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")