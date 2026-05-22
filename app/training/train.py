import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Employee_Salary.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

# Uncomment when running a local MLflow server:
# mlflow.set_tracking_uri("http://127.0.0.1:5555")
mlflow.set_experiment("Employee_Salary_Prediction_Baseline")

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

X = df[[
    "years_experience",
    "age",
    "department",
    "education_level",
    "job_role",
    "performance_score",
]]
y = df["salary"]

categorical_features = ["department", "education_level", "job_role"]
numeric_features = ["years_experience", "age", "performance_score"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

with mlflow.start_run():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mse", mse)

    result = mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    mlflow.register_model(
        model_uri=result.model_uri,
        name="employee-salary-regression-model",
    )

    print(f"Model logged with R2: {r2}")

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl!")
