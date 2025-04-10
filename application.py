from flask import Flask, request, jsonify, render_template
import pandas as pd 
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load trained models safely
model_path = "models/ridge.pkl"
scaler_path = "models/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or Scaler file missing. Ensure 'models/ridge.pkl' and 'models/scaler.pkl' exist.")

ridge_model = pickle.load(open(model_path, "rb"))
standard_scaler = pickle.load(open(scaler_path, "rb"))

# Route for home page
@app.route("/")
def index():
    return render_template("home.html")

# Prediction Route
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extract input data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale the input data
            new_data = np.array([Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]).reshape(1, -1)
            new_data_scaled = standard_scaler.transform(new_data)

            # Predict using the model
            result = ridge_model.predict(new_data_scaled)[0]

            return render_template('home.html', result=result)

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
