#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = '/mnt/data/traffic_dataset_refine.csv'
traffic_data = pd.read_csv(fr"D:\UPES\UPES SEM5\Predictie Analysis\Predictie Analysis Project\Smart Traffic Flow\traffic_dataset_refine.csv")

# Convert Timestamp to datetime
traffic_data['Timestamp'] = pd.to_datetime(traffic_data['Timestamp'])

# Extract additional time-based features
traffic_data['Hour'] = traffic_data['Timestamp'].dt.hour
traffic_data['DayOfWeek'] = traffic_data['Timestamp'].dt.dayofweek
traffic_data['Month'] = traffic_data['Timestamp'].dt.month
traffic_data['DayOfMonth'] = traffic_data['Timestamp'].dt.day

# Create a Weekend feature
traffic_data['Weekend'] = traffic_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Create a Season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

traffic_data['Season'] = traffic_data['Month'].apply(get_season)

# Encode categorical variables
label_encoder = LabelEncoder()
traffic_data['Weather'] = label_encoder.fit_transform(traffic_data['Weather'])
traffic_data['Events'] = traffic_data['Events'].astype(int)

# Split the dataset into features and target variable
X = traffic_data.drop(['Timestamp', 'Traffic Volume'], axis=1)
y = traffic_data['Traffic Volume']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score (Accuracy): {r2}")

# Example: Predict traffic volume for a specific timestamp
def predict_traffic_volume(model, weather, events, road_length, num_lanes, signal_timing, hour, day_of_week, month, day_of_month, weekend, season):
    """
    Predicts traffic volume given specific conditions.
    
    :param model: Trained model
    :param weather: Weather condition as encoded integer
    :param events: 1 if events are present, 0 otherwise
    :param road_length: Length of the road segment
    :param num_lanes: Number of lanes
    :param signal_timing: Traffic signal timing
    :param hour: Hour of the day
    :param day_of_week: Day of the week (0=Monday, 6=Sunday)
    :param month: Month of the year
    :param day_of_month: Day of the month
    :param weekend: 1 if it's a weekend, 0 otherwise
    :param season: Season of the year (0=Winter, 1=Spring, 2=Summer, 3=Fall)
    :return: Predicted traffic volume
    """
    input_data = np.array([[weather, events, road_length, num_lanes, signal_timing, hour, day_of_week, month, day_of_month, weekend, season]])
    return model.predict(input_data)[0]

# Example usage of the function
predicted_volume = predict_traffic_volume(
    rf_model, 
    weather=0,       # e.g., 'Clear' encoded as 0
    events=0,        # No events
    road_length=3.0, # Example road length
    num_lanes=2,     # Example number of lanes
    signal_timing=60, # Example signal timing
    hour=14,         # 2 PM
    day_of_week=2,   # Wednesday
    month=9,         # September
    day_of_month=15, # 15th day
    weekend=0,       # Not a weekend
    season=3         # Fall
)

print(f"Predicted Traffic Volume: {predicted_volume}")


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = '/mnt/data/traffic_dataset_refine.csv'
traffic_data = pd.read_csv(r"D:\UPES\UPES SEM5\Predictie Analysis\Predictie Analysis Project\Smart Traffic Flow\traffic_dataset_refine.csv")

# Convert Timestamp to datetime
traffic_data['Timestamp'] = pd.to_datetime(traffic_data['Timestamp'])

# Extract additional time-based features
traffic_data['Hour'] = traffic_data['Timestamp'].dt.hour
traffic_data['DayOfWeek'] = traffic_data['Timestamp'].dt.dayofweek
traffic_data['Month'] = traffic_data['Timestamp'].dt.month
traffic_data['DayOfMonth'] = traffic_data['Timestamp'].dt.day

# Create a Weekend feature
traffic_data['Weekend'] = traffic_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Create a Season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

traffic_data['Season'] = traffic_data['Month'].apply(get_season)

# Encode categorical variables
label_encoder = LabelEncoder()
traffic_data['Weather'] = label_encoder.fit_transform(traffic_data['Weather'])
traffic_data['Events'] = traffic_data['Events'].astype(int)

# Split the dataset into features and target variable
X = traffic_data.drop(['Timestamp', 'Traffic Volume'], axis=1)
y = traffic_data['Traffic Volume']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the SVM model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score (Accuracy): {r2}")

# Example: Predict traffic volume for a specific timestamp
def predict_traffic_volume(model, scaler, weather, events, road_length, num_lanes, signal_timing, hour, day_of_week, month, day_of_month, weekend, season):
    """
    Predicts traffic volume given specific conditions.
    
    :param model: Trained model
    :param scaler: Scaler object used for standardizing features
    :param weather: Weather condition as encoded integer
    :param events: 1 if events are present, 0 otherwise
    :param road_length: Length of the road segment
    :param num_lanes: Number of lanes
    :param signal_timing: Traffic signal timing
    :param hour: Hour of the day
    :param day_of_week: Day of the week (0=Monday, 6=Sunday)
    :param month: Month of the year
    :param day_of_month: Day of the month
    :param weekend: 1 if it's a weekend, 0 otherwise
    :param season: Season of the year (0=Winter, 1=Spring, 2=Summer, 3=Fall)
    :return: Predicted traffic volume
    """
    input_data = np.array([[weather, events, road_length, num_lanes, signal_timing, hour, day_of_week, month, day_of_month, weekend, season]])
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)[0]

# Example usage of the function
predicted_volume = predict_traffic_volume(
    svm_model, 
    scaler,
    weather=0,       # e.g., 'Clear' encoded as 0
    events=0,        # No events
    road_length=3.0, # Example road length
    num_lanes=2,     # Example number of lanes
    signal_timing=60, # Example signal timing
    hour=14,         # 2 PM
    day_of_week=2,   # Wednesday
    month=9,         # September
    day_of_month=15, # 15th day
    weekend=0,       # Not a weekend
    season=3         # Fall
)

print(f"Predicted Traffic Volume: {predicted_volume}")


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset
file_path = '/mnt/data/traffic_dataset_refine.csv'
traffic_data = pd.read_csv(fr"D:\UPES\UPES SEM5\Predictie Analysis\Predictie Analysis Project\Smart Traffic Flow\traffic_dataset_refine.csv")


# Convert Timestamp to datetime
traffic_data['Timestamp'] = pd.to_datetime(traffic_data['Timestamp'])

# Extract additional time-based features
traffic_data['Hour'] = traffic_data['Timestamp'].dt.hour
traffic_data['DayOfWeek'] = traffic_data['Timestamp'].dt.dayofweek
traffic_data['Month'] = traffic_data['Timestamp'].dt.month
traffic_data['DayOfMonth'] = traffic_data['Timestamp'].dt.day

# Create a Weekend feature
traffic_data['Weekend'] = traffic_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Create a Season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

traffic_data['Season'] = traffic_data['Month'].apply(get_season)

# Encode categorical variables
label_encoder = LabelEncoder()
traffic_data['Weather'] = label_encoder.fit_transform(traffic_data['Weather'])
traffic_data['Events'] = traffic_data['Events'].astype(int)

# Define a binary target variable for classification (e.g., high traffic vs low traffic)
# Here we create a binary target variable based on the median value of 'Traffic Volume'
median_traffic_volume = traffic_data['Traffic Volume'].median()
traffic_data['High Traffic'] = traffic_data['Traffic Volume'].apply(lambda x: 1 if x > median_traffic_volume else 0)

# Split the dataset into features and target variable
X = traffic_data.drop(['Timestamp', 'Traffic Volume', 'High Traffic'], axis=1)
y = traffic_data['High Traffic']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy Score: {accuracy}")
print("Classification Report:")
print(report)

# Example: Predict traffic volume for a specific timestamp
def predict_traffic_class(model, scaler, weather, events, road_length, num_lanes, signal_timing, hour, day_of_week, month, day_of_month, weekend, season):
    """
    Predicts whether traffic volume is high or low given specific conditions.
    
    :param model: Trained model
    :param scaler: Scaler object used for standardizing features
    :param weather: Weather condition as encoded integer
    :param events: 1 if events are present, 0 otherwise
    :param road_length: Length of the road segment
    :param num_lanes: Number of lanes
    :param signal_timing: Traffic signal timing
    :param hour: Hour of the day
    :param day_of_week: Day of the week (0=Monday, 6=Sunday)
    :param month: Month of the year
    :param day_of_month: Day of the month
    :param weekend: 1 if it's a weekend, 0 otherwise
    :param season: Season of the year (0=Winter, 1=Spring, 2=Summer, 3=Fall)
    :return: Predicted class (0=Low Traffic, 1=High Traffic)
    """
    input_data = np.array([[weather, events, road_length, num_lanes, signal_timing, hour, day_of_week, month, day_of_month, weekend, season]])
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)[0]

# Example usage of the function
predicted_class = predict_traffic_class(
    logistic_model, 
    scaler,
    weather=0,       # e.g., 'Clear' encoded as 0
    events=0,        # No events
    road_length=3.0, # Example road length
    num_lanes=2,     # Example number of lanes
    signal_timing=60, # Example signal timing
    hour=14,         # 2 PM
    day_of_week=2,   # Wednesday
    month=9,         # September
    day_of_month=15, # 15th day
    weekend=0,       # Not a weekend
    season=3         # Fall
)

print(f"Predicted Traffic Class: {'High Traffic' if predicted_class == 1 else 'Low Traffic'}")


# In[ ]:




