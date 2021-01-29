# To add a new cell, type ' '
# To add a new markdown cell, type '  [markdown]'

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


pricing = pd.read_csv("Salary.csv")  # reading the data
print(pricing)


X = pricing[["YearsExperience"]]
Y = pricing[["Salary"]]


X = np.reshape(np.array(X), (len(X), 1))
Y = np.reshape(np.array(Y), (len(Y), 1))


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2)  # spliting test and train data


plt.scatter(X, Y)
plt.xlabel("Years Experience (in years)")
plt.ylabel("Salary (in $)")
plt.show()  # scattering the plot


reg = LinearRegression()  # model
reg = reg.fit(X_train, Y_train)  # Fit the linear model


acc = reg.score(X_train, Y_train)  # coefficient of rms error
print(acc)


plt.scatter(X_train, Y_train, color='blue')
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, reg.predict(X_train), color='black')
plt.plot(X_test, reg.predict(X_test), color='green')
plt.xlabel("Years of Experience (in Years)")
plt.ylabel("Salary (in $)")
plt.show()  # final plot with


# predicting few salaries based on years of experience


# predicition no. 1
pred1 = reg.predict(np.array([[13]]))  # 13 years of experience
print(pred1)


# prediction no. 2
pred2 = reg.predict(np.array([[7.3]]))
print(pred2)
