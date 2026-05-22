import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

# Load the trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    years_experience = float(request.form["years_experience"])
    age = int(request.form["age"])
    department = request.form["department"]
    education_level = request.form["education_level"]
    job_role = request.form["job_role"]
    performance_score = float(request.form["performance_score"])

    feature_names = [
        "years_experience",
        "age",
        "department",
        "education_level",
        "job_role",
        "performance_score",
    ]

    features = pd.DataFrame(
        [[
            years_experience,
            age,
            department,
            education_level,
            job_role,
            performance_score,
        ]],
        columns=feature_names,
    )

    prediction = model.predict(features)
    formatted_prediction = f"The predicted salary is £{round(float(prediction[0]), 2):,.2f}"

    return render_template("result.html", prediction=formatted_prediction)


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
