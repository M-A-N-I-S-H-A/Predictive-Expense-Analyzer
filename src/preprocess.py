import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load the data
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', 'data', 'synthetic_expenses.csv')
df = pd.read_csv(file_path)

print("Original Data Sample:")
print(df.head())

# 2. Convert 'Date' to a number (AI can't read '2025-01-01')
# We convert it to 'Ordinal' which is just a count of days
df['Date'] = pd.to_datetime(df['Date'])
df['Date_Ordinal'] = df['Date'].apply(lambda x: x.toordinal())

# 3. Handle Categories (Food, Rent -> 0, 1)
# This is called Label Encoding
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

# 4. Feature Selection
# We only give the AI the numbers: Date, Category, and Amount
features = ['Date_Ordinal', 'Category_Encoded', 'Amount']
X = df[features]

# 5. Scaling (The "Pro" Step)
# This makes sure the 'Amount' doesn't dominate the 'Category' number
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nPreprocessed Data (Scaled):")
print(X_scaled[:5]) # Show first 5 rows

# Save the processed data for the next step (Modeling)
# We convert it back to a DataFrame to save it easily
processed_df = pd.DataFrame(X_scaled, columns=features)
processed_path = os.path.join(current_dir, '..', 'data', 'processed_data.csv')
processed_df.to_csv(processed_path, index=False)

print(f"\nSuccess! Processed data saved at: {processed_path}")