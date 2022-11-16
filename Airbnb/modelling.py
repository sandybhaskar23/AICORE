from tabular_data import load_airbnb
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import datasets, model_selection
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt 





def sgd_modelling(csv,labl):

    ###capture all the features and labels
    (X,y) = load_airbnb(csv,labl)

    #standardise datasets
    X = scale(X)
    y = scale(y)
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.15)
    ###does not like NaN values need to use HistGradientBoostingRegressor() or HistGradientBoostingClassifier()
   
    est = SGDRegressor().fit(X, y).fit(xtrain, ytrain)

    score = est.score(xtrain, ytrain)
    print("R-squared:", score)

    ypred = est.predict(xtest)


    mse = mean_squared_error(ytest, ypred)
    #print("ypred",ypred)
    print("MSE: ", mse)
    print("RMSE: ", mse**(1/2.0))

    x_ax = range(len(ytest))
    plt.plot(x_ax, ytest, label="original")
    plt.plot(x_ax, ypred, label="predicted")
    plt.title("Airbnb house prices and predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    csv = "tabular_data\listing.csv"
    labl = 'Price_Night'
    sgd_modelling(csv,labl)


