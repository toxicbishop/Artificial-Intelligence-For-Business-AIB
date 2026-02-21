"""
=============================================================================
  AI FOR BUSINESS — Week 06
  Topic : AI and Analytics (Google Analytics integration)
  File  : 06.py

  Approach : Create analytics dashboard and predictions
    1. Data Visualization of Google Analytics data.
    2. AI-Powered Predictions using Linear Regression.
    3. Model Performance evaluation.

  Dependencies : pandas, numpy, matplotlib, seaborn, scikit-learn
=============================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
 
# Simulating Google Analytics-like data
np.random.seed(42)
days = pd.date_range(start="2024-01-01", periods=100, freq='D')
sessions = np.random.randint(100, 1000, size=100)  # Random website traffic
bounce_rate = np.random.uniform(30, 80, size=100)  # Bounce rate in %
 
# Creating a DataFrame
df = pd.DataFrame({"Date": days, "Sessions": sessions, "Bounce Rate": bounce_rate})
 
# **1. Data Visualization**
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.lineplot(x="Date", y="Sessions", data=df, marker="o")
plt.title("Daily Website Traffic")
 
plt.subplot(1, 2, 2)
sns.histplot(df["Bounce Rate"], bins=10, kde=True, color="red")
plt.title("Bounce Rate Distribution")
 
plt.show()
 
# **2. AI-Powered Predictions using Linear Regression**
df["Day"] = np.arange(1, 101)  # Convert dates into numerical format
 
# Splitting data
X = df[["Day"]]
y = df["Sessions"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Training Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# Predict future sessions
future_days = np.array([[i] for i in range(101, 111)])  # Next 10 days
predicted_sessions = model.predict(future_days)
 
# Display Predictions
predicted_df = pd.DataFrame({"Day": future_days.flatten(), "Predicted Sessions": predicted_sessions})
print("Predicted Website Traffic for Next 10 Days:\n", predicted_df)
 
# **3. Model Performance**
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE (Lower is better): {mae:.2f}")
 
# Plot Predictions
plt.figure(figsize=(10,5))
sns.lineplot(x=df["Day"], y=df["Sessions"], label="Actual Data")
sns.lineplot(x=predicted_df["Day"], y=predicted_df["Predicted Sessions"], label="Predicted Data", linestyle="dashed")
plt.axvline(x=100, color='red', linestyle='--', label="Today")
plt.title("AI-Powered Traffic Prediction")
plt.legend()
plt.show()