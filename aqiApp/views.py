from django.shortcuts import render
from django.http import HttpResponse
from sklearn.linear_model import LinearRegression
from joblib import load
import numpy as np


def aqipredictor(request):
    return render(request, 'form.html')


def predict_aqi(request):
    # Load the regression model
    #regressor = load('regressor.joblib')

    
    # Importing the libraries

    import numpy as np
    import pandas as pd

# Importing the dataset

    dataset = pd.read_csv("aqi-data.csv")
    X = dataset.iloc[ : , :-1].values
    Y = dataset.iloc[ : , -1].values


# Taking Care of Missing data

    l = X
    for i in range(len(l)):
        for j in range(8):
            if l[i][j] == 'BDL':
                l[i][j] = 'Nan'

    for i in range(len(l)):
        for j in range(8):
            if l[i][j] == ' ':
                l[i][j] = 'Nan'


    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
    imputer.fit(X[:, 2:10])
    X[:,2:10] = imputer.transform(X[:,2:10])

    mean = np.nanmean(Y)
    Y[np.isnan(Y)] = mean

# Encoding Categorical Data

## Encoding the Independent Variable

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder() , [0,1])], remainder="passthrough")
    X = np.array(ct.fit_transform(X))

# Splitting Dataset into Training set and Test set.

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)

# Training the Lasso regression model on the Training set

    from sklearn.linear_model import Lasso
    regressor = Lasso()
    regressor.fit(X_train, Y_train)

# Predicting the result of the test set

    Y_predicted = regressor.predict(X_test)
    np.set_printoptions(precision =2)

# Evaluating the Model Accuracy

    from sklearn.metrics import r2_score
    r2_score(Y_test, Y_predicted)

# Credits:  Harsh Anand (Github: anand-harsh)
    # Get the input values from the form
    city = request.GET['city']
    year = float(request.GET['year'])
    month = request.GET['month']
    so2 = float(request.GET['so2'])
    nh3 = float(request.GET['nh3'])
    nox = float(request.GET['nox'])
    pm10 = float(request.GET['pm10'])
    pm25 = float(request.GET['pm25'])

    # Convert city name to one-hot encoded vector
    city_dict = {'Khammam': [1.0, 0.0, 0.0, 0.0, 0.0], 'Adilabad': [0.0, 1.0, 0.0, 0.0, 0.0], 'Karimnagar': [
        0.0, 0.0, 1.0, 0.0, 0.0], 'Nizamabad': [0.0, 0.0, 0.0, 1.0, 0.0], 'Warangal': [0.0, 0.0, 0.0, 0.0, 1.0]}
    city_values = city_dict[city]

    month_dict = {'January': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Febuary': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'March': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'April': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'May': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'June': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'July': [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'August': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'September': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 'October': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 'November': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'December': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    month_values = month_dict[month]

    # Predict AQI index for the input values
    data_point = [city_values + month_values +
                  [year, so2, nh3, nox, pm10, pm25]]
    Y_predicted = regressor.predict(data_point)
    aqi = round(Y_predicted[0], 2)

    # Return the result to the user
    return render(request, 'result.html', {'aqi': aqi, 'city': city, 'year': year, 'month': month, 'so2': so2, 'nh3': nh3, 'nox': nox, 'pm10': pm10, 'pm25': pm25})

# Credits: Harsh Anand (Github: anand-harsh)
