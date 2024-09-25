from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Load data
melbourne_file_path = r"C:\Users\yashk\Certifications\Machine Learning - KAGGLE\datasets\melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)
print(melbourne_model)


predicted_home_prices = melbourne_model.predict(X)

# print the in-sample score
print(mean_absolute_error(y, predicted_home_prices))

# splitting data into training and validation data for feature and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

print('train and validation values:')
print(train_X,val_X,train_y,val_y)

# get prediction prices on val data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
