import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
import time

# Load the dataset
file_path = "dataset.csv" 
data = pd.read_csv(file_path)

# Step 1: Data Cleaning
start_time = time.time()
print("Step 1: Cleaning Data")

# Drop rows with missing target or features
data.dropna(subset=['popularity'], inplace=True)

# Ensure 'explicit' is binary
data['explicit'] = data['explicit'].astype(int)

# One-hot encoding for 'track_genre'
encoder = OneHotEncoder(sparse_output=False)
genre_encoded = encoder.fit_transform(data[['track_genre']])
genre_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out(['track_genre']))
data = pd.concat([data, genre_df], axis=1)
data.drop(columns=['track_genre'], inplace=True)

print(f"Data cleaning completed in {time.time() - start_time:.2f} seconds.\n")

# Step 2: Feature Selection
print("Step 2: Feature Selection")
features = [
    'danceability', 'energy', 'loudness', 'acousticness',
    'instrumentalness', 'valence', 'tempo', 'duration_ms', 'explicit'
] + list(encoder.get_feature_names_out(['track_genre']))
target = 'popularity'

# Drop rows with missing values in selected features
data.dropna(subset=features, inplace=True)

# Splitting the data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Scaling Features
print("Step 3: Scaling Features")
scaler = StandardScaler()
X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])

# Step 4: Model Training with Optimizations
print("Step 4: Model Training")
model = RandomForestRegressor(
    n_estimators=50,  
    random_state=42,
    n_jobs=-1, 
    verbose=1   
)

# Fit model and measure time
train_start = time.time()
model.fit(X_train, y_train)
train_end = time.time()
print(f"Model training completed in {train_end - train_start:.2f} seconds.\n")

# Step 5: Predictions and Evaluation
print("Step 5: Evaluating Model")

# Check for NaN in X_test or predictions
if X_test.isnull().sum().sum() > 0:
    print("Error: X_test contains NaN values.")
    exit()

y_pred = model.predict(X_test)
y_pred = np.nan_to_num(y_pred)  # Replace NaNs with 0 in predictions

if np.isnan(y_test).sum() > 0:
    print("Error: y_test contains NaN values.")
    exit()

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR^2: {r2:.2f}")

# Step 6: Analyze Misclassified Samples
print("Step 6: Analyzing Misclassified Samples")
errors = abs(y_test - y_pred)
misclassified_indices = errors.nlargest(5).index
misclassified_samples = data.iloc[misclassified_indices]
print("\nMisclassified Samples:")
print(misclassified_samples[['track_name', 'artists', 'popularity', 'danceability', 'valence', 'tempo']])

# Step 7: Feature Importance
print("\nStep 7: Feature Importance")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance)

# Save results and features importance
misclassified_samples.to_csv("misclassified_samples.csv", index=False)
importance.to_csv("feature_importance.csv", index=False)

print("\nScript completed.")
