from tabular_data import load_airbnb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score,accuracy_score
from sklearn.preprocessing import scale
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder


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
        self.select_model= np.empty((11,0))
        self.model_selected ={}
        

    def sgd_modelling(self,csv,labl):

        ###capture all the features and labels
        (X,y) = load_airbnb(csv,labl,'classification')

        X = OneHotEncoder().fit_transform(X)

        #standardise datasets
        #sc = StandardScaler()
        #X = sc.fit_transform(X)
        #y = sc.fit_transform(y)
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
       
            LogisticRegression : self.hyparamtyp1_dist ,
            RandomForestClassifier :self.hyparamtyp2_dist,
            KNeighborsClassifier :self.hyparamtyp3_dist ,
            MLPClassifier : self.hyparamtyp4_dist,
            AdaBoostClassifier : self.hyparamtyp5_dist  
              
        }

    def hyperparameters(self):

        ##set some standard value where ML algo have overlapping parameters name
        ##note sharing same name does not denote same behaviour across each model
        regularisation = np.array([0.1,0.2,1,2,3,4],dtype=float)
        penalty = np.array(['l1','l2'])
        warm_start = [True]
        n_estimators = np.array([2,4,8,16,32,64,1024])
        max_depth = np.array([1,2,3,4,5,6,7,8,9,10])
        n_neighbors = np.arange(1,21)

        ##chnage solver form default due to L1 penalty being seen in data
        ##chnage max_iter from default since more were needed from 100->1000->5000
        ##anything more than a 1000 iteration is an issue 
        self.hyparamtyp1_dist = {
            'C' : regularisation,
            'penalty' : penalty,
            'warm_start' : warm_start,
            'max_iter' : [1000,5000],
            'solver' : ['liblinear']       

        }

        self.hyparamtyp2_dist = {
            'n_estimators' : n_estimators,
            'max_depth' : max_depth,
            'bootstrap' : [True]
        }

        self.hyparamtyp3_dist = {
            'n_neighbors' : n_neighbors,
            'metric' : ['euclidean', 'cityblock']
        }

        self.hyparamtyp4_dist= {
            'solver' : ['sgd'],
            'learning_rate':['adaptive'],
            'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,)],
            'activation': ['relu', 'tanh', 'logistic', 'identity'],
            'max_iter' : [500,1000,5000]
        }

        self.hyparamtyp5_dist = {
            'n_estimators' : n_estimators,
            'learning_rate' : [1,2,3,4,5],
            }




    def tune_classification_model_hyperparameters(self,models,train_split=None):

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

            ##get precision recal,F1 and accuracy scores for training and test 
            y_pred = tuner.predict(ts['xtrain'])
            y_pred_test = tuner.predict(ts['xtest'])
            p,r,f1,ac = self.get_prfac(ts['ytrain'],y_pred,'micro')
            pt,rt,f1t,act = self.get_prfac(ts['ytest'],y_pred_test,'micro')

            print(f"Precision:{p}\tRecall:{r}\tF1:{f1}\tAccuracy:{ac}")
            
            ####join the param to form a unique key
            sep = '_'
            _k = sep.join(str(tuner.best_params_[x]) for x in sorted(tuner.best_params_))     
            self.best_hyper[type(mod()).__name__] = {_k : tuner.best_score_}

            columns = np.array([[type(mod()).__name__,_k,p,r,f1,ac,pt,rt,f1t,act,tuner.best_score_]])
            #print(columns.ndim, columns.shape)
            #print(self.select_model.ndim, self.select_model.shape)
            self.select_model = np.append(self.select_model, columns.transpose(),axis=1)

            self.save_model(type(mod()).__name__,tuner,models[mod],self.best_hyper)


    def get_prfac(self, ytrain, y_pred, avg='micro'):
            p = precision_score(ytrain, y_pred, average=avg, zero_division=0)
            r = recall_score(ytrain, y_pred, average=avg,zero_division=0)
            f1 = f1_score(ytrain, y_pred, average=avg,zero_division=0)
            ac = accuracy_score(ytrain, y_pred)
        
            return(p,r,f1,ac)

    
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
        #print(mn_rmse_loss,bst_hypeindex)
        ###this is a possible limitation where the index have been explicitly defined as beeing assocaited to a data type
        self.model_selected = {
                'Best_model' : self.select_model[0][bst_hypeindex],
                'Parameters' : self.select_model[1][bst_hypeindex],
                'RMSE' : mn_rmse_loss,
                'Precision_train' : self.select_model[2][bst_hypeindex],
                'Recall_train' : self.select_model[3][bst_hypeindex],
                'F1_train' : self.select_model[4][bst_hypeindex],
                'Accuracy_train' : self.select_model[5][bst_hypeindex],
                'Precision_test' : self.select_model[6][bst_hypeindex],
                'Recall_test' : self.select_model[7][bst_hypeindex],
                'F1_test' : self.select_model[8][bst_hypeindex],
                'Accuracy_test' : self.select_model[9][bst_hypeindex],
                }

        return(self.model_selected)
            
        


def evaluate_all_models(csv,labl):

    mgs = ModellingGridSearch()
    mgs.sgd_modelling(csv,labl)
    mgs.define_models()
  
    #mgs.custom_tune_regression_model_hyperparameters(mgs.models,mgs.train_split,mgs.hyparam_dist)

    mgs.tune_classification_model_hyperparameters(mgs.models,mgs.train_split)
    
    return(mgs.find_best_model())
    



        

if __name__ == "__main__":

    csv = "tabular_data\listing.csv"
    labl = 'Category'

    bst_mod = evaluate_all_models(csv,labl)
    for key,val in bst_mod.items():
        print(f"{key} : {val}")
    
    
