from tabular_data import load_airbnb
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, model_selection
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt 
import pandas as pd



class ModellingGridSearch:

    def __init__(self):

        self.best_hyper = {}
        self.valid_loss = []
        self.train_split ={}
        self.best_hyp={}
        self.data  = {}
        

    def sgd_modelling(self,csv,labl):

        ###capture all the features and labels
        (X,y) = load_airbnb(csv,labl)

        #standardise datasets
        X = scale(X)
        y = scale(y)
        xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, y, test_size=0.3)
        ###does not like NaN values need to use HistGradientBoostingRegressor() or HistGradientBoostingClassifier()

        xvalid, xtest, yvalid, ytest = model_selection.train_test_split(xtest, ytest, test_size=0.5)
    
        self.train_split = {
            'xtrain' : xtrain,
            'xtest' : xtest,
            'ytrain' : ytrain,
            'ytest' : ytest,
            'xvalid' :xvalid,
            'yvalid' : yvalid
        }

        
    def plot_data(self):
        x_ax = range(len(ytest))
        plt.plot(x_ax, ytest, label="original")
        plt.plot(x_ax, ypred, label="predicted")
        plt.title("Airbnb house prices and predicted data")
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc='best',fancybox=True, shadow=True)
        plt.grid(True)
        plt.show()



    def define_models(self):

        self.models = [
       
            ExtraTreesRegressor,
            BaggingRegressor,
            RandomForestRegressor,
              
        ]

    def hyperparameters(self):


        self.hyparam_dist = {
            'max_samples' : np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            'n_estimators' : np.array([2,4,8,16,32,64,1024])
        }

    def custom_tune_regression_model_hyperparameters(self,models,train_split=None,hyp=None):


        np.random.seed(2)
        ##shorthand notation for dict
        if train_split is not None:
            ts = self.train_split
        else:
            ts = train_split


        for mod in models:
            best_hyp = []
            rmse_loss =[]
            self.data[mod.__class__.__name__] = {}
            for ms in np.nditer(hyp['max_samples']):
                for ne in np.nditer(hyp['n_estimators']):

                    est = mod(n_estimators = int(ne), max_samples = float(ms),bootstrap=True).fit(ts['xtrain'], ts['ytrain'])

                    ytrain_pred = est.predict(ts['xtrain'])
                    yvalid_pred = est.predict(ts['xvalid'])
                    ytest_pred = est.predict(ts['xtest'])

                    ##works out the loss of function score    
                    train_mse = mean_squared_error(ts['ytrain'],ytrain_pred)
                    valid_mse = mean_squared_error(ts['yvalid'],yvalid_pred)
                    test_mse = mean_squared_error(ts['ytest'],ytest_pred)

                    valrmse_loss=(valid_mse**(1/2.0))
                    self.data[mod.__class__.__name__].update( {
                            f"{ms}_{ne}" : {
                            'Validation_MSE_Loss' : valid_mse,
                            'Training_MSE_Loss' : train_mse,
                            'Test_MSE_Loss' : test_mse,
                            'Validation_RMSE_Loss' : valrmse_loss
                        }}
                    )

                    best_hyp.append(f"{ms}_{ne}" )
                    rmse_loss.append(valrmse_loss)

            ##lower the RMSEq the better
            mn_rmse_loss = min(rmse_loss)
            bst_hypeindex = rmse_loss.index(mn_rmse_loss)
            #print(f"Class: {mod}, hyperparameter(mx_smp_n-esti): {best_hyp[bst_hypeindex]}, min RMSE LOSS : {mn_rmse_loss}")
            self.best_hyper[mod] = {best_hyp[bst_hypeindex] : mn_rmse_loss}


    def pretty_data(self):

        df = pd.DataFrame(self.data)
        df.to_csv("all_score.csv")

        bstdf = pd.DataFrame(self.best_hyper)
        print(bstdf)


        

if __name__ == "__main__":

    csv = "tabular_data\listing.csv"
    labl = 'Price_Night'
    mgs = ModellingGridSearch()
    mgs.sgd_modelling(csv,labl)
    mgs.define_models()
    mgs.hyperparameters()
    mgs.custom_tune_regression_model_hyperparameters(mgs.models,mgs.train_split,mgs.hyparam_dist)
    mgs.pretty_data()


