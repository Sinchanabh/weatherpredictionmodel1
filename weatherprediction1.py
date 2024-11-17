import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



# Step 1: Generate synthetic weather data (for demonstration purposes)
# Replace this with your actual weather dataset (e.g., weather_data.csv)
np.random.seed(42)

dates = pd.date_range('2024-11-04', periods=100, freq='D')
temperature = np.random.uniform(10, 30, size=100)
humidity = np.random.uniform(50, 90, size=100)
pressure = np.random.uniform(1000, 1020, size=100)
wind_speed = np.random.uniform(0, 10, size=100)
temperature_next_day = temperature + np.random.uniform(-2, 2, size=100)

# Create DataFrame
weather_data = pd.DataFrame({
    'date': dates,
    'temperature': temperature,
    'humidity': humidity,
    'pressure': pressure,
    'wind_speed': wind_speed,
    'temperature_next_day': temperature_next_day
})

# Step 2: Preprocessing the data
X = weather_data[['temperature', 'humidity', 'pressure', 'wind_speed']]  # Features
y = weather_data['temperature_next_day']  # Target

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Regressor Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Step 6: Function for predicting next day's temperature based on user input
def predict_weather(temperature_input, humidity_input, pressure_input, wind_speed_input):
    # Prepare the input data (reshape to match model's input format)
    input_data = np.array([[temperature_input, humidity_input, pressure_input, wind_speed_input]])
    
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

# Step 7: Taking user inputs for prediction
print("Enter the current weather information to predict tomorrow's temperature:")

# User input
temperature_input = float(input("Enter current temperature (°C): "))
humidity_input = float(input("Enter current humidity (%): "))
pressure_input = float(input("Enter current atmospheric pressure (hPa): "))
wind_speed_input = float(input("Enter current wind speed (m/s): "))

# Step 8: Predict next day's temperature based on user input
predicted_temperature = predict_weather(temperature_input, humidity_input, pressure_input, wind_speed_input)

# Output prediction
print(f"\nPredicted temperature for the next day: {predicted_temperature:.2f}°C")


# Visualization
plt.plot(y_test.values, label='Actual Temperature')
plt.plot(y_pred, label='Predicted Temperature')
plt.legend()
plt.show()
