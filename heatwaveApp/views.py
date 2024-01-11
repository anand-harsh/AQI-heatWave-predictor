from django.shortcuts import render
from django.http import HttpResponse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np


def heatwavepredictor(request):
    return render(request, 'heat-wave.html')

def predict_heatwave(request):
        
        
        # Load the regression model
        #regressor = load('model.joblib')
        
            # Importing the libraries
        import numpy as np
        import pandas as pd

# Importing the dataset
        dataset = pd.read_csv("heatwave-data.csv")

        X = dataset.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
        Y = dataset.iloc[:, -2].values

# Taking Care of Missing data
        l = X
        for i in range(len(l)):
            for j in range(10):
                if l[i][j] == 'BDL':
                    l[i][j] = 'Nan'

        for i in range(len(l)):
            for j in range(10):
                if l[i][j] == ' ':
                    l[i][j] = 'Nan'

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer.fit(X[:, 3:11])
        X[:, 3:11] = imputer.transform(X[:, 3:11])
        mean = np.nanmean(Y)
        Y[np.isnan(Y)] = mean

# Encoding Categorical Data
# Encoding the Independent Variable
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1])], remainder="passthrough")
        X = np.array(ct.fit_transform(X))

# Splitting Dataset into Training set and Test set.
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1)

# Training the Multiple Linear regression model on the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)

# Predicting the Test set results
        Y_test_predicted = regressor.predict(X_test)
        np.set_printoptions(precision=2)

# Evaluating the Model Accuracy
        from sklearn.metrics import r2_score
        r2_score(Y_test, Y_test_predicted)

# Developed By Team Codex
# Credits: Rishit Kumar , Harsh Anand (Github: anand-harsh)

        # Get the input values from the form
        city=request.GET['city']
        year=float(request.GET['year'])
        month=request.GET['month']
        rainfall=float(request.GET['rainfall'])
        mintemp=float(request.GET['mintemp'])
        maxtemp=float(request.GET['maxtemp'])
        minhumidity=float(request.GET['minhumidity'])
        maxhumidity=float(request.GET['maxhumidity'])
        minwindspeed=float(request.GET['minwindspeed'])
        maxwindspeed=float(request.GET['maxwindspeed'])
        
        # Convert city name to one-hot encoded vector
        city_dict = {'Khammam': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'ADILABAD': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Karimnagar': [0.0, 1.0, 0.0 ,0.0 ,0.0, 0.0], 'Nizamabad': [0.0, 0.0 ,0.0 ,1.0 ,0.0, 0.0], 'WARANGAL R': [0.0 ,0.0 ,0.0, 0.0 ,1.0 ,0.0], 'WARANGAL U': [0.0, 0.0 ,0.0 ,0.0, 0.0, 1.0]}
        city_values = city_dict[city]

        month_dict={'January':[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0 ,0.0, 0.0, 0.0, 0.0], 'Febuary':[0.0, 0.0 ,0.0, 1.0 ,0.0, 0.0, 0.0 ,0.0 ,0.0 ,0.0, 0.0 ,0.0], 'March':[0.0 ,0.0 ,0.0 ,0.0, 0.0 ,0.0 ,0.0, 1.0, 0.0, 0.0 ,0.0, 0.0], 'April':[1.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0], 'May':[0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'June':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0], 'July':[0.0, 0.0 ,0.0 ,0.0, 0.0 ,1.0 ,0.0, 0.0, 0.0 ,0.0 ,0.0 ,0.0], 'August':[0.0 ,1.0, 0.0 ,0.0 ,0.0 ,0.0, 0.0,0.0, 0.0, 0.0 ,0.0 ,0.0], 'September':[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0, 0.0, 0.0 ,0.0, 1.0], 'October':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 'November':[0.0 ,0.0, 0.0, 0.0, 0.0, 0.0,0.0 ,0.0, 0.0, 1.0, 0.0, 0.0], 'December':[0.0, 0.0 ,1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0 ,0.0]}
        month_values = month_dict[month]

        # Predict AQI index for the input values
        data_point = [city_values + month_values + [year, rainfall, mintemp, maxtemp, minhumidity, maxhumidity, minwindspeed, maxwindspeed]]
        Y_predicted = regressor.predict(data_point)
        heatindex = round(Y_predicted[0], 2)

        # Return the result to the user
        return render(request, 'heatwave-result.html', {'heatindex': heatindex, 'city':city, 'year':year, 'month':month, 'rainfall':rainfall, 'mintemp':mintemp, 'maxtemp':maxtemp, 'maxhumidity':maxhumidity, 'minhumidity':minhumidity, 'minwindspeed':minwindspeed, 'maxwindspeed':maxwindspeed})

# Developed By Team Codex
# Credits: Harsh Anand (Github: anand-harsh), Rishit Kumar