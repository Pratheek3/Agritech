from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend
@app.route('/')
@app.route('/rainfall.html')
def rainfall():
    return app.send_static_file('rainfall.html')

@app.route('/crop.html')
def crop():
    return app.send_static_file('crop.html')

@app.route('/yield.html')
def yield_page():
    return app.send_static_file('yield.html')
# Load models
try:
    rain_model = joblib.load("annual_rainfall_model.pkl")
    crop_model = joblib.load("crop_recommendation_model.pkl")
    yield_model = joblib.load("yield_prediction_model.pkl")
    crop_enc = joblib.load("crop_encoder.pkl")
    season_enc = joblib.load("season_encoder.pkl")
    state_enc = joblib.load("state_encoder.pkl")
except FileNotFoundError as e:
    print(f"Error: Model or encoder file not found: {e}")
    raise

@app.route('/')
def home():
    return jsonify({"message": "Welcome to Crop Yield Prediction API"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_type = request.form['form_type']
        
        if form_type == 'rainfall':
            season = season_enc.transform([request.form['season'].strip()])[0]
            state = state_enc.transform([request.form['state'].strip()])[0]
            crop = crop_enc.transform([request.form['crop'].strip()])[0]
            area = float(request.form['area'])
            input_data = np.array([[season, state, crop, area]])
            prediction = rain_model.predict(input_data)[0]
            return jsonify({"result": f"Predicted Annual Rainfall: {round(prediction, 2)} mm"})

        elif form_type == 'crop':
            season = season_enc.transform([request.form['season'].strip()])[0]
            state = state_enc.transform([request.form['state'].strip()])[0]
            area = float(request.form['area'])
            rainfall = float(request.form['rainfall'])
            fertilizer = float(request.form['fertilizer'])
            pesticide = float(request.form['pesticide'])
            production = float(request.form['production'])
            yield_val = float(request.form['yield'])
            crop_year = int(request.form['crop_year'])
            input_data = np.array([[season, state, area, rainfall, fertilizer, pesticide, production, yield_val, crop_year]])
            pred_code = crop_model.predict(input_data)[0]
            prediction = crop_enc.inverse_transform([pred_code])[0]
            return jsonify({"result": f"Recommended Crop: {prediction}"})

        elif form_type == 'yield':
            crop = crop_enc.transform([request.form['crop'].strip()])[0]
            state = state_enc.transform([request.form['state'].strip()])[0]
            season = season_enc.transform([request.form['season'].strip()])[0]
            production = float(request.form['production'])
            area = float(request.form['area'])
            fertilizer = float(request.form['fertilizer'])
            pesticide = float(request.form['pesticide'])
            input_data = np.array([[crop, state, season, production, area, fertilizer, pesticide]])
            prediction = yield_model.predict(input_data)[0]
            return jsonify({"result": f"Predicted Yield: {round(prediction, 2)}"})

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 400

    return jsonify({"error": "Invalid form type"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')