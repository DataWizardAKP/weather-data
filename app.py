from flask import Flask, request, render_template
import pickle
import numpy as np
import torch
import os
from model import MLPRegressor

# ---- Load bundle: model + scaler + feature order ----
BUNDLE_PATH = "feelslike_one.pkl"
with open(BUNDLE_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]          # nn.Module
scaler = bundle["scaler"]        # StandardScaler
feature_cols = bundle["feature_cols"]  # list of column names used in training

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    # Show empty page (no prediction yet)
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form in the SAME ORDER as feature_cols
        # Names in your index.html should match these keys:
        # temperature_celsius, wind_mph, wind_degree, pressure_mb,
        # precip_mm, humidity, cloud, uv_index, gust_mph
        x_list = []
        for key in feature_cols:
            val = request.form.get(key, type=float)
            if val is None:
                return render_template(
                    "index.html",
                    prediction_value="Input error: missing value for {}".format(key)
                )
            x_list.append(val)

        x = np.array([x_list], dtype=np.float32)         # shape (1, 9)
        xs = scaler.transform(x).astype(np.float32)      # scale with saved scaler

        # Inference with torch model
        with torch.no_grad():
            y = model(torch.from_numpy(xs)).numpy().ravel()[0]

        return render_template("index.html", prediction_value=round(float(y), 2))

    except Exception as e:
        return render_template("index.html", prediction_value=f"Error: {e}")

if __name__ == "__main__":
    # IMPORTANT: bind to $PORT and 0.0.0.0
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
