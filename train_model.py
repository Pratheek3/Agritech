import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("crop_yield.csv")

# Strip extra spaces from categorical columns
df['Crop'] = df['Crop'].str.strip()
df['Season'] = df['Season'].str.strip()
df['State'] = df['State'].str.strip()

# Make a copy
df_copy = df.copy()

# Label Encoding
crop_enc = LabelEncoder()
season_enc = LabelEncoder()
state_enc = LabelEncoder()

df_copy['Crop'] = crop_enc.fit_transform(df_copy['Crop'])
df_copy['Season'] = season_enc.fit_transform(df_copy['Season'])
df_copy['State'] = state_enc.fit_transform(df_copy['State'])

# Features and Target for Crop Recommendation
features = ['Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Production', 'Yield', 'Crop_Year']
X_crop = df_copy[features]
y_crop = df_copy['Crop']

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X_crop, y_crop):
    X_train = X_crop.iloc[train_idx]
    X_test = X_crop.iloc[test_idx]
    y_train = y_crop.iloc[train_idx]
    y_test = y_crop.iloc[test_idx]

crop_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    class_weight="balanced",
    random_state=42
)
crop_model.fit(X_train, y_train)

# Evaluate Crop Model
y_pred = crop_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Crop Classification Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=crop_enc.classes_))

# Features and Target for Yield Prediction
X_yield = df_copy[['Crop', 'State', 'Season', 'Production', 'Area', 'Fertilizer', 'Pesticide']]
y_yield = df_copy['Yield']

X_train, X_test, y_train, y_test = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)
yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
yield_model.fit(X_train, y_train)

# Evaluate Yield Model
y_pred = yield_model.predict(X_test)
print("Yield Prediction R² Score:", r2_score(y_test, y_pred))


# Features and Target for Annual Rainfall
X_rain = df_copy[['Season', 'State', 'Crop', 'Area']]
y_rain = df_copy['Annual_Rainfall']

X_train, X_test, y_train, y_test = train_test_split(X_rain, y_rain, test_size=0.2, random_state=42)
rain_model = RandomForestRegressor(n_estimators=100, random_state=42)
rain_model.fit(X_train, y_train)

# Evaluate Rainfall Model
y_pred = rain_model.predict(X_test)
print("Annual Rainfall R² Score:", r2_score(y_test, y_pred))

# Save models and encoders
joblib.dump(crop_model, "crop_recommendation_model.pkl")
joblib.dump(yield_model, "yield_prediction_model.pkl")
joblib.dump(rain_model, "annual_rainfall_model.pkl")
joblib.dump(crop_enc, "crop_encoder.pkl")
joblib.dump(season_enc, "season_encoder.pkl")
joblib.dump(state_enc, "state_encoder.pkl")