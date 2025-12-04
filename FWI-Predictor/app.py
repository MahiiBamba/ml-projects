from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load pre-trained scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("ridge.pkl", "rb") as f:
    model = pickle.load(f)


def fwi_to_risk(fwi_value: float):
    """
    Categorize the predicted Fire Weather Index (FWI) into risk levels.
    """
    if fwi_value <= 5:
        return ("Low", "Conditions are safe -- fire risk is minimal.", "low")
    elif fwi_value <= 10:
        return ("Moderate", "Be alert -- fires can start under dry conditions.", "moderate")
    elif fwi_value <= 20:
        return ("High", "Caution! Fire spread is likely under current conditions.", "high")
    else:
        return ("Extreme", "Danger! High chance of fire ignition and rapid spread.", "extreme")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form input
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        DC = float(request.form.get("DC"))
        ISI = float(request.form.get("ISI"))
        BUI = float(request.form.get("BUI"))
        Region = int(request.form.get("Region"))

        features = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, Region]])
        features_scaled = scaler.transform(features)
        fwi_pred = float(model.predict(features_scaled)[0])

        risk_label, advisory, risk_class = fwi_to_risk(fwi_pred)

        return render_template(
            "home.html",
            prediction=f"{fwi_pred:.2f}",
            risk_label=risk_label,
            advisory=advisory,
            risk_class=risk_class
        )

    except Exception as e:
        return render_template(
            "home.html",
            prediction="—",
            risk_label="Error",
            advisory=f"Invalid input or server error: {str(e)}",
            risk_class="error"
        )


if __name__ == "__main__":
    app.run(debug=True)
