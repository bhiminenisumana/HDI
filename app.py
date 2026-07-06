# ============================================================
# app.py — Flask Backend for HDI Prediction Web Application
# ============================================================
# This application loads a pre-trained Linear Regression model
# (hdi_model.pkl) and serves a web form where users can enter
# 4 socio-economic indicators to predict a country's
# Human Development Index (HDI) score.
#
# Author: Capstone Project — A Comprehensive Measure of Well-Being
# ============================================================

import os
import pickle
import numpy as np
from flask import Flask, render_template, request

# ------------------------------------------------------------
# STEP 1 — Initialize the Flask application
# ------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------
# STEP 2 — Load the trained Linear Regression model
# ------------------------------------------------------------
# The model was trained in Google Colab (Epics 3-7) and saved
# using pickle. We load it once when the server starts, so we
# don't reload it on every single request (much faster).
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "hdi_model.pkl")

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    print(f"[INFO] Model loaded successfully from: {MODEL_PATH}")
except FileNotFoundError:
    # If the model file is missing, we let the app start anyway
    # (so the homepage still loads) but predictions will fail
    # gracefully with a clear error message instead of crashing.
    model = None
    print(f"[ERROR] Model file not found at: {MODEL_PATH}")
    print("[ERROR] Please make sure 'hdi_model.pkl' is inside the 'model/' folder.")

# ------------------------------------------------------------
# STEP 3 — Define acceptable input ranges for basic sanity checks
# ------------------------------------------------------------
# These are not hard restrictions from the dataset, just realistic
# bounds to catch obviously mistyped values (e.g. negative years).
VALID_RANGES = {
    "life_expectancy": (0, 100),      # years
    "mean_yr_school": (0, 25),        # years
    "expec_yr_school": (0, 25),       # years
    "gross_inc_percap": (0, 250000),  # USD
}


def validate_and_parse_inputs(form_data):
    """
    Validates and converts raw form input (strings) into a clean
    list of floats in the exact order the model expects.

    Parameters
    ----------
    form_data : werkzeug.datastructures.ImmutableMultiDict
        The raw POST form data from the HTML form.

    Returns
    -------
    tuple (values, error_message)
        values -> list of 4 floats if valid, otherwise None
        error_message -> None if valid, otherwise a user-friendly string
    """
    field_order = ["life_expectancy", "mean_yr_school",
                   "expec_yr_school", "gross_inc_percap"]

    field_labels = {
        "life_expectancy": "Life Expectancy",
        "mean_yr_school": "Mean Years of Schooling",
        "expec_yr_school": "Expected Years of Schooling",
        "gross_inc_percap": "GNI per Capita",
    }

    values = []

    for field in field_order:
        raw_value = form_data.get(field, "").strip()

        # Check 1: field must not be empty
        if raw_value == "":
            return None, f"{field_labels[field]} is required. Please enter a value."

        # Check 2: field must be a valid number
        try:
            numeric_value = float(raw_value)
        except ValueError:
            return None, f"{field_labels[field]} must be a valid number (you entered '{raw_value}')."

        # Check 3: field must fall within a realistic range
        min_val, max_val = VALID_RANGES[field]
        if not (min_val <= numeric_value <= max_val):
            return None, (f"{field_labels[field]} should be between {min_val} and {max_val}. "
                          f"You entered {numeric_value}.")

        values.append(numeric_value)

    return values, None


# ------------------------------------------------------------
# STEP 4 — Home page route
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Renders the home page with the empty input form."""
    return render_template("index.html", prediction=None, error=None, form_values={})


# ------------------------------------------------------------
# STEP 5 — Prediction route
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles form submission:
    1. Validates the 4 input fields.
    2. Converts them into the format the model expects.
    3. Runs the prediction.
    4. Renders the result back on the same page.
    """
    # Keep the submitted values so the form can be re-populated
    # even if there's a validation error (better user experience).
    form_values = {
        "life_expectancy": request.form.get("life_expectancy", ""),
        "mean_yr_school": request.form.get("mean_yr_school", ""),
        "expec_yr_school": request.form.get("expec_yr_school", ""),
        "gross_inc_percap": request.form.get("gross_inc_percap", ""),
    }

    # Guard clause: model failed to load at startup
    if model is None:
        return render_template(
            "index.html",
            prediction=None,
            error="Prediction unavailable: model file not found on the server.",
            form_values=form_values,
        )

    # Validate and parse the submitted form inputs
    values, error_message = validate_and_parse_inputs(request.form)

    if error_message:
        # Invalid input -> show the error, keep the form filled in
        return render_template(
            "index.html",
            prediction=None,
            error=error_message,
            form_values=form_values,
        )

    try:
        # The model expects a 2D array: [[feature1, feature2, feature3, feature4]]
        # in the SAME order used during training:
        # life_expectancy, mean_yr_school, expec_yr_school, gross_inc_percap
        input_array = np.array(values).reshape(1, -1)

        # Run the prediction
        predicted_hdi = model.predict(input_array)[0]

        # Clip prediction into the valid HDI range [0, 1] just in case
        # the linear model slightly overshoots at extreme input values.
        predicted_hdi = float(np.clip(predicted_hdi, 0, 1))

        # Round to 3 decimal places, matching how HDI is usually reported
        predicted_hdi = round(predicted_hdi, 3)

        return render_template(
            "index.html",
            prediction=predicted_hdi,
            error=None,
            form_values=form_values,
        )

    except Exception as e:
        # Catch-all safety net: never let the app crash on the user.
        print(f"[ERROR] Prediction failed: {e}")
        return render_template(
            "index.html",
            prediction=None,
            error="Something went wrong while generating the prediction. Please try again.",
            form_values=form_values,
        )


# ------------------------------------------------------------
# STEP 6 — Run the Flask development server
# ------------------------------------------------------------
if __name__ == "__main__":
    # debug=True gives helpful error pages during development.
    # Set debug=False before deploying to production.
    app.run(debug=True, host="0.0.0.0", port=5000)