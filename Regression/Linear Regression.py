import matplotlib as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()
# what is in the data set
#['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
print(diabetes.keys())
diabetes_X=diabetes.data
print(diabetes_X)

diabetes_X_train=diabetes_X[:-30]# features
diabetes_X_test=diabetes_X[-30:]

diabetes_Y_train=diabetes.target[:-30] #lable
diabetes_Y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()
model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predicted=model.predict(diabetes_X_test)

#main sqare error
print("Mean squared error is :", mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))

print("weight:",model.coef_)
print("Intercept :" ,model.intercept_)
import matplotlib as plt
import matplotlib.pyplot as plt
plt.scatter(diabetes_X_test, diabetes_Y_test)
plt.plot(diabetes_X_test, diabetes_Y_predicted)
plt.show()
