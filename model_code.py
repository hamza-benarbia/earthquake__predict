import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your data
data = pd.read_csv("database.csv")

# Extract relevant columns
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# Convert Date and Time to Timestamp
timestamp = []
date_time_formats = ['%m/%d/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ']

for d, t in zip(data['Date'], data['Time']):
    parsed = False
    for fmt in date_time_formats:
        try:
            ts = pd.to_datetime(d + ' ' + t, format=fmt)
            timestamp.append(ts.timestamp())
            parsed = True
            break
        except ValueError:
            pass
    if not parsed:
        print(f"Error processing timestamp: Unknown datetime string format, unable to parse: {d} {t}")
        timestamp.append(np.nan)

# Ensure the length of the timestamp list matches the DataFrame length
data['Timestamp'] = timestamp[:len(data)]

# Drop rows with missing or NaN timestamp values
final_data = data.dropna(subset=['Timestamp'])

# Drop unnecessary columns
final_data = final_data.drop(['Date', 'Time'], axis=1)

# Split the data into features (X) and target variables (y)
X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]

# Handle the case where the dataset is too small for the split
if len(X) == 0:
    raise ValueError("Dataset is too small. Adjust parameters or provide a larger dataset.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# Save the trained model
joblib.dump(reg, 'random_forest_model.h5')
