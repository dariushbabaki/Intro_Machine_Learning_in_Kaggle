# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *
print("Setup Complete")

# Import the train_test_split function and uncomment
# from _ import _

# fill in and uncomment
from sklearn.model_selection import train_test_split

#Assuming the data is read from a file and variables X and y are defined.
import pandas as pd
file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(file_path)

X = home_data[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = home_data['SalePrice']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Check your answer
step_1.check()

# Step 1: Split Your Data
# The lines below will show you a hint or the solution.
# step_1.hint()
# step_1.solution()


# Step 2: Specify and Fit the Model.

# You imported DecisionTreeRegressor in your last exercise
# and that code has been copied to the setup code above. So, no need to
# import it again

# Specify the model
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

# Check your answer
step_2.check()


# step_2.hint()
# step_2.solution()


# Step 3: Make Predictions with Validation data
# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# Check your answer
step_3.check()


# step_3.hint()
# step_3.solution()

# Inspect your predictions and actual values from validation data.

# print the top few validation predictions
print(val_predictions[:5])

# print the top few actual prices from validation data
print(val_y[:5])

# Step 4: Calculate the Mean Absolute Error in Validation Data

# Make predictions for the entire dataset (both training and validation)
predictions = iowa_model.predict(X)

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)

# uncomment following line to see the validation mae
#print(val_mae)

# Check your answer
step_4.check()

# step 4.hint()
# step 4.solution()
