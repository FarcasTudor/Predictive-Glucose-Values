import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load the model and explicitly specify the custom_objects if needed
model = tf.keras.models.load_model('glucose_forecasting_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Mock input data for prediction (last 12 glucose readings spaced 5 minutes apart)
data = {
    'datetime': pd.date_range('2024-05-07 00:00', periods=12, freq='5min'),
    'glucose': [81,83,85,87,88,88,87,86,85,84,85,86]
}

input_df = pd.DataFrame(data)
input_df.set_index('datetime', inplace=True)

scaler = joblib.load('scaler.gz')
input_df['glucose'] = scaler.transform(input_df[['glucose']])

# Convert to numpy array suitable for model prediction
input_sequence = np.array([input_df['glucose'].values])
input_sequence = np.expand_dims(input_sequence, -1)  # Reshape for the model

# Make predictions
predictions = model.predict(input_sequence)
predictions = scaler.inverse_transform(predictions)  # Inverse transform to original scale

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(input_df.index, scaler.inverse_transform(input_df[['glucose']]), label='Actual Glucose Levels', color='blue', marker='o')
future_times = pd.date_range(start=input_df.index[-1] + pd.Timedelta(minutes=5), periods=5, freq='5min')
plt.plot(future_times, predictions[0], label='Predicted Glucose Levels', color='red', marker='o')
plt.xlabel('Datetime')
plt.ylabel('Glucose Level')
plt.title('Actual vs Predicted Glucose Levels')
plt.legend()
plt.grid(True)
plt.show()
