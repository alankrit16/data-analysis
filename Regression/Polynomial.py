import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

"""no point of test train splitting for this dataset"""

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=5)
x_poly = poly_regressor.fit_transform(x)

linear_regressor2 = LinearRegression()
linear_regressor2.fit(x_poly,y)

#visualize linear reg model
plt.scatter(x,y,color='red')
plt.plot(x,linear_regressor.predict(x),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#visualize poly reg model
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red')
plt.plot(x_grid,linear_regressor2.predict(poly_regressor.fit_transform(x_grid)),color='blue')
plt.title('Truth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predict with linear regression
linear_regressor.predict(6.5)

#predict with polynomial regression
linear_regressor2.predict(poly_regressor.fit_transform(6.5))
