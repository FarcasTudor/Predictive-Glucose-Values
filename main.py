import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(data.columns[0], axis=1)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

data = preprocess_data('glucose.csv')

# Normalize the data
scaler = MinMaxScaler()
data['glucose'] = scaler.fit_transform(data[['glucose']])

# Function to create sequences and targets
def create_sequences(data, sequence_length, forecast_length):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length - forecast_length + 1):
        sequences.append(data[i:(i + sequence_length)].values)
        targets.append(data[(i + sequence_length):(i + sequence_length + forecast_length)].values)
    return np.array(sequences), np.array(targets)

# Create sequences
sequence_length = 12  # last 60 minutes
forecast_length = 5   # next 30 minutes
X, y = create_sequences(data['glucose'], sequence_length, forecast_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1])  # Output layer for predicting 5 future values
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Print the model summary
model.summary()

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

# Plot the glucose levels
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['glucose'], label='Glucose Levels', color='blue')
plt.xlabel('Datetime')
plt.ylabel('Glucose Level')
plt.title('Glucose Levels Over Time')
plt.legend()
plt.show()

# Plot the predictions
plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(y_test):], y_test[:, 0], label='Actual Glucose Levels', color='blue')
plt.plot(data.index[-len(y_test):], predictions[:, 0], label='Predicted Glucose Levels', color='red')
plt.xlabel('Datetime')
plt.ylabel('Glucose Level')
plt.title('Actual vs Predicted Glucose Levels')
plt.legend()
plt.show()

# Save the model
model.save('glucose_forecasting_model1.keras')
joblib.dump(scaler, 'scaler.gz')
