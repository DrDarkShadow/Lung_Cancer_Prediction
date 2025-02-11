from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Define the model path and load the trained model
model_path = os.path.join(os.getcwd(), 'lung_cancer_model.pkl')
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print("Error loading model:", e)

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Retrieve and process form inputs
            gender = 1 if request.form.get('gender') == '1' else 0
            age = float(request.form.get('age'))
            smoking = 1 if request.form.get('smoking') == '1' else 0
            yellow_fingers = 1 if request.form.get('yellow_fingers') == '1' else 0
            anxiety = 1 if request.form.get('anxiety') == '1' else 0
            peer_pressure = 1 if request.form.get('peer_pressure') == '1' else 0
            chronic_disease = 1 if request.form.get('chronic_disease') == '1' else 0
            fatigue = 1 if request.form.get('fatigue') == '1' else 0
            allergy = 1 if request.form.get('allergy') == '1' else 0
            wheezing = 1 if request.form.get('wheezing') == '1' else 0
            alcohol_consuming = 1 if request.form.get('alcohol_consuming') == '1' else 0
            coughing = 1 if request.form.get('coughing') == '1' else 0
            shortness_of_breath = 1 if request.form.get('shortness_of_breath') == '1' else 0
            swallowing_difficulty = 1 if request.form.get('swallowing_difficulty') == '1' else 0
            chest_pain = 1 if request.form.get('chest_pain') == '1' else 0

            # Create a feature array in the order expected by your model
            features = [
                gender,
                age,
                smoking,
                yellow_fingers,
                anxiety,
                peer_pressure,
                chronic_disease,
                fatigue,
                allergy,
                wheezing,
                alcohol_consuming,
                coughing,
                shortness_of_breath,
                swallowing_difficulty,
                chest_pain
            ]
            features = np.array(features).reshape(1, -1)

            # Debug: Print input features
            print("Input Features:", features)

            # Ensure the model is loaded
            if model is None:
                raise Exception("Model not loaded correctly.")

            # Make prediction using the loaded model
            result_proba = model.predict_proba(features)[0][1]
            prediction_text = f"There is a {result_proba * 100:.2f}% chance of lung cancer."

            # Debug: Print prediction probability result
            print("Prediction Probability Result:", result_proba)
        except Exception as e:
            prediction_text = "Error during prediction: " + str(e)
            print("Error:", e)

    return render_template('index.html', prediction=prediction_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)