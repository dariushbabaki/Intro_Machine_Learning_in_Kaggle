# Code you have previously used to load data
import pandas as pd
# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *
print("Setup Complete")

# Step 1: Specify Prediction Target
# print the list of columns in the dataset to find the name of the prediction target
print(home_data.columns)

y = home_data['SalePrice']

# Check your answer
step_1.check()

# The lines below will show you a hint or the solution.
# step_1.hint()
# step_1.solution()

# Step 2: Create X
# Create a list of the predictive features
feature_columns = ['LotArea', 'Year Built', '1st FlrSF', '2nd FlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Create the DataFrame X
X = home_data[feature_columns]

# Check your answer 
step_2.check()

# The lines below will show you a hint or the solution.
# step_2.hint()
# step_2.solution()

# Review Data
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())

# Step 3: Specify and Fit Model
from sklearn.tree import DecisionTreeRegressor

# Define the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)

# Check your answer
step_3.check()

# The lines below will show you a hint or the solution.
# step_3.hint()
# step_3.solution()

# Step 4: Make Predictions
# Make predictions with the model's predict command using X as the data. Save the results to a variable called predictions
predictions = iowa_model.predict(X)

# Calculate Mean Absolute Error
from sklearn.metrics import mean_absolute_error

# Prediction for educational data
predicted_home_prices = iowa_model.predict(X)

# Calculating the mean absolute error
mae = mean_absolute_error(y, predicted_home_prices)
print("Mean Absolute Error:", mae)

# Check your answer
step_4.check()

# Check your answer
# step_4.hint()
# step_4.solution()

# Think About Your Results
# Use the head method to compare the top few predictions to the actual home values (in y) for those same homes. Anything surprising?
print("Actual Sale Prices:", y.head().values)
print("Predicted Sale Prices:", predictions[:5])
