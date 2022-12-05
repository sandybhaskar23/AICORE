from tabular_data import load_airbnb
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from pathlib import Path

import joblib
import json
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ModellingGridSearch:

    def __init__(self):

        self.best_hyper = {}
        self.valid_loss = []
        self.train_split ={}
        self.best_hyp={}
        self.data  = {}
        self.select_model= np.empty((3,0))
        self.model_selected ={}
        

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

        self.hyperparameters()

        self.models = {
       
            ExtraTreesRegressor : self.hyparamtyp1_dist ,
            BaggingRegressor : self.hyparamtyp1_dist ,
            RandomForestRegressor : self.hyparamtyp1_dist ,
            GradientBoostingRegressor : self.hyparamtyp2_dist ,
            DecisionTreeRegressor : self.hyparamtyp3_dist
              
        }

    def hyperparameters(self):

        ##set some standard value where ML algo have overlapping parameters name
        ##note sharing same name does not denote same behaviour across each model
        n_estimators = np.array([2,4,8,16,32,64,1024])
        learning_rate = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        max_samples  =  np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        max_depth = np.array([1,2,3,4,5,6,7,8,9,10])

        self.hyparamtyp1_dist = {
            'max_samples' : max_samples,
            'n_estimators' : n_estimators,
            'bootstrap' : [True]
        }

        self.hyparamtyp2_dist={
            'n_estimators' : n_estimators,
            'learning_rate' : learning_rate
        }

        self.hyparamtyp3_dist = {
            'max_depth' : max_depth            
        }



    def tune_regression_model_hyperparameters(self,models,train_split=None):

          ##shorthand notation for dict
        if train_split is not None:
            ts = self.train_split
        else:
            ts = train_split
        

        for mod in models:

            tuner = GridSearchCV(estimator = mod(), param_grid= models[mod])
            tuner.fit(ts['xtrain'], ts['ytrain'])
            print(type(mod()).__name__)
            print(tuner.best_params_)
            print(tuner.best_score_)
            ####join the param to fomr a unique key
            sep = '_'
            _k = sep.join(str(tuner.best_params_[x]) for x in sorted(tuner.best_params_))     
            self.best_hyper[type(mod()).__name__] = {_k : tuner.best_score_}

            columns = np.array([[type(mod()).__name__,_k,tuner.best_score_]])
            print(columns.ndim, columns.shape)
            print(self.select_model.ndim, self.select_model.shape)
            self.select_model = np.append(self.select_model, columns.transpose(),axis=1)
            #self.select_model = np.append(self.select_model, [['type(mod()).__name__'],[_k],[tuner.best_score_]])

            #self.select_model['bst_model'].append(type(mod()).__name__)
            #self.select_model['best_hyp'].append(_k)
            #self.select_model['rmse_loss'].append(tuner.best_score_)
            self.save_model(type(mod()).__name__,tuner,models[mod],self.best_hyper)




    def custom_tune_regression_model_hyperparameters(self,models,train_split=None,hyp=None):


        np.random.seed(2)
        ##shorthand notation for dict
        if train_split is not None:
            ts = self.train_split
        else:
            ts = train_split

        ##loop through models
        for mod in models:
            best_hyp = []
            rmse_loss =[]
            self.data[type(mod()).__name__] = {}
            ##now loop through hyperparameters
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
                    self.data[type(mod()).__name__].update( {
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
            self.best_hyper[type(mod()).__name__] = {best_hyp[bst_hypeindex] : mn_rmse_loss}


    def pretty_data(self):

        if self.data is not None:
            self.df = pd.DataFrame(self.data)
            #df.to_csv("all_score.csv")

        self.bstdf = pd.DataFrame(self.best_hyper)
        #print(self.bstdf)

    def get_path(self,nme):

        modir = Path(f'models/regression/{nme}')
        modir.mkdir(parents=True,exist_ok=True)

        return modir

    def save_model(self,nme, model,hyp, metric):
                
        modir = self.get_path(nme)
             
        ##write model details to file
        joblib.dump(model,f"{modir}\\{nme}.joblib")
        ###jsonify the parameters
        
        jpath = modir/ "hyperparameters.json"
        ##numpyencoder deal with json.dumps limitations todealing with numpy arrays
        jpath.write_text(json.dumps(hyp,cls=NumpyEncoder))

        ##try data frame could use json.dumps
        hyperparam = pd.DataFrame(metric)
        hyperparam.to_json(f"{modir}\metrics.json")

        #self.pretty_data()
    def find_best_model(self):

        print(self.select_model)


        ##lower the RMSEq the better
        mn_rmse_loss = min(self.select_model[-1])
        bst_hypeindex = np.where(self.select_model[-1]==mn_rmse_loss)[0][0]  #self.select_model[-1].index(mn_rmse_loss)
        print(mn_rmse_loss,bst_hypeindex)
        self.model_selected[self.select_model[0][bst_hypeindex]] = {self.select_model[1][bst_hypeindex] : mn_rmse_loss}

        return(self.model_selected)
            
        


def evaluate_all_models(csv,labl):

    mgs = ModellingGridSearch()
    mgs.sgd_modelling(csv,labl)
    mgs.define_models()
  
    #mgs.custom_tune_regression_model_hyperparameters(mgs.models,mgs.train_split,mgs.hyparam_dist)

    mgs.tune_regression_model_hyperparameters(mgs.models,mgs.train_split)
    
    return(mgs.find_best_model())
    



        

if __name__ == "__main__":

    csv = "tabular_data\listing.csv"
    labl = 'Price_Night'

    bst_mod = evaluate_all_models(csv,labl)

    print(bst_mod)??now document
    
