
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
folder = Path(__file__).parent

file_path = folder / "Tomato_Sandy_Clay_Loam_3000_Dataset_Stage1_Radiance_Synthetic.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Stage1_Initial_Model"
)

# 2. Separate input and output
X = df.drop(columns=["water_required"])
y = df["water_required"]

# 3. Identify categorical and numerical columns
categorical_columns = [
    "soil_type",
    "crop_type",
    "crop_stage"
]

numerical_columns = [
    "soil_moisture_before",
    "temperature_before",
    "humidity_before",
    "rainfall_forecast_before",
    "wind_speed_before",
    "solar_radiance_before"
]

# 4. Preprocess categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_columns)
    ],
    remainder="passthrough"
)

# 5. Create the ML model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# 6. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Make predictions
predictions = model.predict(X_test)

# 9. Evaluate the model
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2 Score:", r2_score(y_test, predictions))

# 10. Save the trained model
joblib.dump(model, folder / "initial_irrigation_model.pkl")

print("Initial ML model trained and saved successfully!")
