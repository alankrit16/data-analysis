import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")
#for x-->y
x = dataset.iloc[:,:1].values
y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,
                                                 random_state=0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predict salary 
y_pred = regressor.predict(x_test)

#visualizing the training set result
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Sal vs Exp (training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#visualize the testing set result
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_test,y_pred, color = 'blue')
plt.title('Sal vs Exp (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()