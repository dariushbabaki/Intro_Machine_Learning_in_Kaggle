# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import * 
print("Setup Complete")

# Step 1: Loading Data
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Call line below with no argument to check that you've loaded the data correctly
step_1.check()


# Step 2: Review The Data
# Print summary statistics in next line
home_data.describe()

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = round(home_data['LotArea'].mean())

# As of today, how old is the newest home (current year - the date in which it was built)?
import datetime
current_year = datetime.datetime.now().year
newest_home_age = current_year - home_data['YearBuilt'].max()

# Check your answers
step_2.check()

# Uncomment these lines if you want hints or solutions
# step_2.hint()
# step_2.solution()
