from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)  # <-- use =, not ==

# Load model and scaler
with open("best_regressor.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route('/')
def fun():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    try:
        # If your HTML still has gender, you can read it but ignore it
        # _gender = float(request.form['gender'])

        age        = float(request.form['age'])
        height     = float(request.form['height'])
        weight     = float(request.form['weight'])
        duration   = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp  = float(request.form['body_temp'])

        # Must match training features: ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
        features = [age, height, weight, duration, heart_rate, body_temp]
        arr = np.array([features])  # shape (1, 6)

        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]

        return render_template(
            'index.html',
            prediction_text=f"Predicted calories burned: {pred:.2f}"
        )
    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error during prediction: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)
