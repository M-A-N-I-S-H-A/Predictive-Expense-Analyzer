import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

def generate_finance_data(n_rows=1000):
    categories = ['Food', 'Rent', 'Transport', 'Entertainment', 'Utilities', 'Shopping']
    data = []
    start_date = datetime(2025, 1, 1)

    for i in range(n_rows):
        date = start_date + timedelta(days=random.randint(0, 365))
        category = random.choice(categories)
        
        # Normal spending logic
        if category == 'Rent':
            amount = 1500  
        elif category == 'Food':
            amount = np.random.normal(20, 5) 
        else:
            amount = np.random.uniform(10, 100)
            
        data.append([date, category, round(amount, 2), "Normal"])

    # ADDING ANOMALIES (To impress recruiters)
    for _ in range(20):
        # Huge Splurge
        data.append([start_date + timedelta(days=random.randint(0, 365)), 
                     'Shopping', random.uniform(2000, 5000), "Anomaly_Splurge"])
        # Double Bill
        date_val = start_date + timedelta(days=random.randint(0, 365))
        data.append([date_val, 'Utilities', 150.00, "Anomaly_Double"])
        data.append([date_val, 'Utilities', 150.00, "Anomaly_Double"])

    df = pd.DataFrame(data, columns=['Date', 'Category', 'Amount', 'Label'])
    return df.sort_values(by='Date')

# --- FIXING THE FILE PATHS ---
# 1. This finds the folder where this script is saved
current_folder = os.path.dirname(os.path.abspath(__file__))

# 2. This goes "UP" one level and looks for the 'data' folder
project_root = os.path.dirname(current_folder)
data_folder = os.path.join(project_root, 'data')

# 3. Create the data folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# 4. Define the final file path
file_path = os.path.join(data_folder, 'synthetic_expenses.csv')

# 5. Generate and Save
df_finance = generate_finance_data()
df_finance.to_csv(file_path, index=False)

print(f"Success! Your file is saved at: {file_path}")