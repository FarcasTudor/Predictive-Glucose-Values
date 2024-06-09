from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*", "headers": "Content-Type"}})

# Load the pre-trained model and scaler
model = tf.keras.models.load_model('glucose_forecasting_model1.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
scaler = joblib.load('scaler.gz')

@app.route('/predict', methods=['POST'])
@cross_origin(origins='*', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data)
    # Extract glucose values and timestamps from the request data
    glucose_values = data['glucose']
    print(glucose_values)
    timestamps = pd.to_datetime(data['datetime'])

    # Create DataFrame from the received data
    input_df = pd.DataFrame({'datetime': timestamps, 'glucose': glucose_values})
    input_df.set_index('datetime', inplace=True)

    # Scale the glucose values using the loaded scaler
    input_df['glucose'] = scaler.transform(input_df[['glucose']])

    # Prepare input for the model
    input_sequence = np.array([input_df['glucose'].values])
    input_sequence = np.expand_dims(input_sequence, -1)  # Reshape for the model

    # Make predictions
    predictions = model.predict(input_sequence)
    predictions = scaler.inverse_transform(predictions)  # Inverse transform to original scale

    print("PREDICTIONS in json: " + str(predictions[0].tolist()))
    response = {
        'predicted_glucose_levels': predictions[0].tolist(),
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
