import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'aqi_model.pkl')

# Load Model with Memory Mapping to prevent MemoryError
if os.path.exists(MODEL_PATH):
    # mmap_mode='r' reads from disk instead of RAM
    model = joblib.load(MODEL_PATH, mmap_mode='r')
    print("Model loaded successfully!")
else:
    model = None
    print("Run train_model.py first!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model not found."
    
    try:
        # User input from form
        inputs = [float(x) for x in request.form.values()]
        # Predict
        prediction = model.predict([inputs])
        res = round(prediction[0], 2)
        
        # AQI Category Logic
        status = "Good" if res <= 50 else "Moderate" if res <= 100 else "Unhealthy"
        
        return render_template('index.html', 
                               prediction_text=f'Predicted AQI: {res}',
                               status=f'Air Condition: {status}')
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)