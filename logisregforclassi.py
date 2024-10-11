# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge

# Step 1: Load the California housing dataset
housing = fetch_california_housing()
X = housing.data  # Features
y = housing.target  # Target (house prices)

# Step 2: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Instantiate the KernelRidge model with desired hyperparameters
krr_model = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)  # RBF kernel

# Step 4: Fit the model to the training data
krr_model.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = krr_model.predict(X_test)

# Step 6: Evaluate the model's performance
krr_mse = mean_squared_error(y_test, y_pred)

# Step 7: Print the results
print("Kernel Ridge Regression Mean Squared Error (RBF Kernel):", krr_mse)
