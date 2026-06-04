import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# =========================
# MLflow CONFIG (MUST BE FIRST)
# =========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-risk-model")

# =========================
# CONFIG
# =========================
# mlflow.set_experiment("credit-risk-model")
  # Ensure tracking directory exists

# =========================
# LOAD DATA
# =========================
data_path = "data/processed/processed_data.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Processed data not found at {data_path}. Please run feature engineering first!")

df = pd.read_csv(data_path)
print(f"Data loaded: {df.shape}")

# =========================
# DROP UNNECESSARY COLUMNS
# =========================
drop_cols = [
    "TransactionId", "BatchId", "AccountId", "SubscriptionId", 
    "CustomerId", "TransactionStartTime", "FraudResult"
]

df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("is_high_risk", axis=1)
y = df["is_high_risk"]

print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# COLUMN GROUPS
# =========================
categorical_cols = [col for col in [
    "CurrencyCode", "CountryCode", "ProviderId", "ProductCategory", 
    "ChannelId", "PricingStrategy"
] if col in X.columns]

numeric_cols = [col for col in [
    "Amount", "Value", "transaction_hour", "transaction_day", 
    "transaction_month", "transaction_year", "total_transaction_amount",
    "avg_transaction_amount", "transaction_count", "std_transaction_amount"
] if col in X.columns]

print(f"Numeric features: {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

# =========================
# PREPROCESSOR
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)

# =========================
# EVALUATION
# =========================
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    
    print(f"\n{model_name} Performance:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return metrics

# =========================
# TRAIN MODELS
# =========================

# Model 1: Logistic Regression
with mlflow.start_run(run_name="LogisticRegression"):
    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    lr_model.fit(X_train, y_train)
    metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 1000})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(lr_model, "model")
    
    print(" Logistic Regression run logged")

# Model 2: Random Forest
with mlflow.start_run(run_name="RandomForest"):
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    rf_model.fit(X_train, y_train)
    metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    mlflow.log_params({"model_type": "RandomForest", "n_estimators": 100})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rf_model, "model")
    
    print(" Random Forest run logged")

print("\n Training completed! Run `mlflow ui` to view results.")
