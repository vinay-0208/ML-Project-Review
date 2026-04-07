import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv(r"C:\ML Project\StreetLight_Project\streetlight_energy_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Features & Target
X = df.drop("energy_consumption_kwh", axis=1)
y = df["energy_consumption_kwh"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Multiple Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []
best_model = None
best_model_name = ""
best_r2 = -999

n = len(X_test)
p = X_test.shape[1]

# Training & Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)

    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(f"\n{name}")
    print("R2:", r2)
    print("Adjusted R2:", adj_r2)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAE:", mae)

    results.append([name, r2, adj_r2, mse, rmse, mae])

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# Results table
results_df = pd.DataFrame(results, columns=[
    "Model", "R2", "Adjusted R2", "MSE", "RMSE", "MAE"
])

print("\n===== MODEL COMPARISON TABLE =====")
print(results_df.sort_values(by="R2", ascending=False))

# Accuracy graph
plt.figure()
plt.bar(results_df["Model"], results_df["R2"])
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.show()

print("Best Model Selected:", best_model_name)

# Feature Importance / Coefficients
if hasattr(best_model, "feature_importances_"):
    fi = pd.DataFrame({
        "Feature": X.columns,
        "Importance": best_model.feature_importances_
    })
    print(fi.sort_values(by="Importance", ascending=False))

elif hasattr(best_model, "coef_"):
    fi = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": best_model.coef_
    })
    print(fi.sort_values(by="Coefficient", ascending=False))

# Save model
joblib.dump(best_model, r"C:\ML Project\StreetLight_Project\energy_model.pkl")
print("Best Model Saved Successfully")

# ================= TIME SERIES =================
df["datetime"] = pd.date_range(
    start="2024-01-01",
    periods=len(df),
    freq="h"
)

df = df.sort_values("datetime")

ts_df = df[["datetime", "energy_consumption_kwh"]].copy()
ts_df.set_index("datetime", inplace=True)

ts_df["lag_1"] = ts_df["energy_consumption_kwh"].shift(1)
ts_df["lag_2"] = ts_df["energy_consumption_kwh"].shift(2)
ts_df["lag_3"] = ts_df["energy_consumption_kwh"].shift(3)
ts_df["rolling_mean_3"] = ts_df["energy_consumption_kwh"].rolling(3).mean()

ts_df.dropna(inplace=True)

X_ts = ts_df[["lag_1", "lag_2", "lag_3", "rolling_mean_3"]]
y_ts = ts_df["energy_consumption_kwh"]

split = int(len(ts_df) * 0.8)

X_train_ts = X_ts[:split]
X_test_ts = X_ts[split:]
y_train_ts = y_ts[:split]
y_test_ts = y_ts[split:]

ts_model = RandomForestRegressor(n_estimators=200, random_state=42)
ts_model.fit(X_train_ts, y_train_ts)

ts_pred = ts_model.predict(X_test_ts)

ts_r2 = r2_score(y_test_ts, ts_pred)
ts_mse = mean_squared_error(y_test_ts, ts_pred)
ts_rmse = np.sqrt(ts_mse)
ts_mae = mean_absolute_error(y_test_ts, ts_pred)

print("\nTime Series Results")
print("R2:", ts_r2)
print("MSE:", ts_mse)
print("RMSE:", ts_rmse)
print("MAE:", ts_mae)

# ================= VISUALIZATIONS =================

# Linear Regression Predictions
lin_reg_model = models["Multiple Linear Regression"]
lin_reg_pred = lin_reg_model.predict(X_test)

# Graph 1: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lin_reg_pred, alpha=0.6, edgecolors='k')

min_val = min(y_test.min(), lin_reg_pred.min())
max_val = max(y_test.max(), lin_reg_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.title('Actual vs Predicted Energy Consumption')
plt.xlabel('Actual Energy (kWh)')
plt.ylabel('Predicted Energy (kWh)')
plt.grid(True)
plt.show()

# Graph 2: Feature vs Target
plt.figure(figsize=(8, 6))
plt.scatter(X_test['light_intensity'], y_test, alpha=0.5)

m, b = np.polyfit(X_test['light_intensity'], y_test, 1)
plt.plot(X_test['light_intensity'], m * X_test['light_intensity'] + b)

plt.title('Light Intensity vs Energy Consumption')
plt.xlabel('Light Intensity')
plt.ylabel('Energy (kWh)')
plt.grid(True)
plt.show()
