import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data and create variables
dataset = pd.read_csv('C:/Users/USER-PC/Desktop/Code Folder/Sales Prediction Model/Advertising Sales.csv')
X = dataset.iloc[:,1:4]
y = dataset['Sales']
X_train, X_test, y_train, y_test = train_test_split(X,y)

#fit the model for prediction
lin = LinearRegression()
lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)

#create coefficient
coef = lin.coef_
components = pd.DataFrame(zip(X.columns, coef), columns = ['component', 'value'])
intercept_row = pd.DataFrame([{'component': 'intercept', 'value': lin.intercept_}])

import numpy as np 

intercept_value = lin.intercept_.item() if hasattr(lin.intercept_, 'item') else lin.intercept_
intercept_row = pd.DataFrame([{'component': 'intercept', 'value': intercept_value}])
components = pd.concat([components, intercept_row], ignore_index= True)

