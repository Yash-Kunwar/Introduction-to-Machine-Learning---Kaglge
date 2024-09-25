import pandas as pd

melbourne_file_path = r"C:\Users\yashk\Certifications\Machine Learning - KAGGLE\datasets\melb_data.csv"

melbourne_data = pd.read_csv(melbourne_file_path)

# drop null rows - increases accuracy
melbourne_data = melbourne_data.dropna(axis=0)

# gives a description of the csv(comma seperated values) file
print(melbourne_data.describe())

# setting up the prediction target(y)
y = melbourne_data.Price

# choosing the features(X) for prediction 
melbourne_features = ['Rooms', 'Bathroom',
                      'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

print('Making prdeictions for the following five houses')
# extracting first few values from the dataframe using .head()
print(X.head())

# using scikit-learn to make the model
from sklearn.tree import DecisionTreeRegressor

# define model, random state is the identity that python gives to randomisation to ensure same result each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# fit model - capture patterns
melbourne_model.fit(X,y)
print(melbourne_model)

# predict prices
print('the predictions are:-')
print(melbourne_model.predict(X.head()))

# evaluate with the dataframe - rows 3,4,6,8,9
