import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "Employee_Salary.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

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

model.fit(X_train, y_train)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl!")
