"""
train.py — Egypt Real Estate Appraiser
Training pipeline: loads cleaned data → features → trains best model → saves model.pkl
Run: python train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "cleaned_data.csv")

print("=" * 60)
print("  Egypt Real Estate Appraiser — Training Pipeline")
print("=" * 60)
print(f"\n[1/5] Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df):,} records with {df.shape[1]} features.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Feature engineering...")

def extract_governorate(loc):
    if loc == "Other":
        return "Other"
    return loc.split(",")[-1].strip()

type_map = {
    "Apartment": "Apartment", "Chalet": "Chalet", "Villa": "Villa",
    "Townhouse": "Townhouse", "Duplex": "Duplex", "Twin House": "Twin House",
    "Penthouse": "Penthouse", "iVilla": "Villa",
}

df["governorate"]   = df["location"].apply(extract_governorate)
df["property_type"] = df["type"].map(type_map).fillna("Other")
df["price_m"]       = df["price"] / 1_000_000

le_gov  = LabelEncoder()
le_type = LabelEncoder()
le_pay  = LabelEncoder()
df["gov_enc"]  = le_gov.fit_transform(df["governorate"])
df["type_enc"] = le_type.fit_transform(df["property_type"])
df["pay_enc"]  = le_pay.fit_transform(df["payment_method"])

features = ["size_sqm", "bedrooms_num", "bathrooms", "gov_enc", "type_enc", "pay_enc"]
X = df[features]
y = df["price_m"]
print(f"      Features: {features}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Train/Test Split
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Train & Evaluate All Three Models
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Training models...")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    ),
}

results = {}
print(f"\n{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE':>8} {'CV R²':>8}")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    mape  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

    results[name] = {
        "model": model, "y_pred": y_pred,
        "MAE (M EGP)": round(mae, 3), "RMSE (M EGP)": round(rmse, 3),
        "R² Score": round(r2, 4), "MAPE (%)": round(mape, 2),
        "CV R² (5-fold)": round(cv_r2, 4),
    }
    print(f"{name:<22} {mae:>8.3f} {rmse:>8.3f} {r2:>8.4f} {mape:>7.1f}% {cv_r2:>8.4f}")

print("-" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Save Best Model
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving best model...")

best_name  = max(results, key=lambda n: results[n]["R² Score"])
best_model = results[best_name]["model"]

model_bundle = {
    "model"          : best_model,
    "model_name"     : best_name,
    "le_gov"         : le_gov,
    "le_type"        : le_type,
    "le_pay"         : le_pay,
    "features"       : features,
    "governorates"   : list(le_gov.classes_),
    "property_types" : list(le_type.classes_),
    "payment_methods": list(le_pay.classes_),
    "metrics"        : {k: v for k, v in results[best_name].items()
                        if k not in ("model", "y_pred")},
}

save_path = os.path.join(os.path.dirname(__file__), "model.pkl")
joblib.dump(model_bundle, save_path)

print(f"\n  ✅ Best model: {best_name}")
print(f"     R² Score : {results[best_name]['R² Score']}")
print(f"     MAE      : {results[best_name]['MAE (M EGP)']}M EGP")
print(f"     Saved to : {save_path}")
print("\n" + "=" * 60)
print("  Training complete! Run `streamlit run app.py` to launch.")
print("=" * 60)
