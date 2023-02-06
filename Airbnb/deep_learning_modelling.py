from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score 
from tabular_data import AirbnbNightlyPriceImageDataset 

from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 
from statistics import mean 
import numpy as np
import pandas as pd
import time
import torch
import yaml

##

import json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types
        Deals with json.dumps limitations todealing with numpy arrays
        Also addtion to deal with functions 
    
     """
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
        elif isinstance(obj,(RandomForestClassifier , LogisticRegression)):
            return str(obj)

        return json.JSONEncoder.default(self, obj)


class ModellingSearch:

    def __init__(self,config='nn_config.yaml'):

        ##if batch size batches the feature size then this previes the mat1 and mat2 errors
        self.batchsize = 10
        self.data_loader = {}
        ##load the config file
        self.configfile= config
        self.config = self.get_nn_config(config)
        self.global_mse = list 
        self.model_metrics= {}
        ##this is for 7 rows of data on a 2D array
        self.select_model= np.empty((7,0))
        print (self.config)
  
        
        

    def split_modelling(self,csv,labl):
                
        """ Split the data ready for classificaton

            Args:
                csv                 : Data for splitting
                Labl                : What are we trying to predict
        
            
            Returns:
                Binds split data to class 
        
        """
      
        ###capture all the features and labels
        dataset = AirbnbNightlyPriceImageDataset(csv,labl,'deep')

        ###need to split dataset into train and test set from torsh.utils.data classes 
        train_dataset, test_dataset = random_split( dataset, [0.7,0.3])

        ##split train further in to train and validation
        train_dataset, validation_dataset = random_split( train_dataset, [0.8,0.2])

        self.data_loader = {
            'train' : DataLoader(dataset = train_dataset,batch_size=self.batchsize, shuffle=True),
            'test' : DataLoader(dataset = test_dataset,batch_size=self.batchsize, shuffle=True),
            'validation'  : DataLoader(dataset = validation_dataset,batch_size=self.batchsize, shuffle=True),  
        }
      


    def train(self,  loadertype = None, epochs =9):
        """
        REads the loadertype and passes to NN class to then calculate the loss, R2, inference latency and training duration

        Arg:
            loadertype: dictionary caling the appropriate dataset type e.g.  validation,train,test 
            epoch : Capacity or  number of full pass through the training model
        
        Return:
            Binds dictionary of metric to class 
        
        """

        start_time = time.time()
        #self.model = LinearRegression(self.batchsize)
        self.model = NN(self.batchsize,self.config)

        ##set model optimiser 
        optimiser = torch.optim.SGD(self.model.parameters(),lr=0.000099)

        writer = SummaryWriter()        
        print(f'Loaderytpe is {loadertype}')
        batch_index = 0
        overall_sqrt = []
        overall_rsq = []
        overall_inference_latency = []
        
        for epoch in range(epochs):
            epoch_mse =[]        
            epoch_rsq = []
            pred_time = []
            for batch in self.data_loader[loadertype]:
                features, labels = batch
                #print(features)
                features = features.type(torch.FloatTensor)
                pred_stime = time.time()
                prediction = self.model(features)
                pred_etime = time.time()
                ##append to caculate the average latency for each loader type
                pred_time.append(pred_etime - pred_stime)
                ##flatten the prediction to make the target size and the the input size to match. 
                prediction = prediction.reshape(-1)
                #print(torch.tensor(labels).shape)                         
                loss = F.mse_loss(prediction.float(), labels.float())                  
                #get MSE to work out average later                
                epoch_mse.append(loss.item())                
                ##get r_squared to work out average later
                #print(type(labels),type(prediction))
                #print("SOL",labels.detach().numpy(),prediction.detach().numpy(),"EOL")
                epoch_rsq.append(r2_score(labels.detach().numpy(),prediction.detach().numpy()))
                loss.backward()              
                optimiser.step()
                ##re initalise this optimiser.  Do not want it informing the next batch
                optimiser.zero_grad()
                
                writer.add_scalar('loss',loss.item(),batch_index)
                batch_index += 1
            ##work out the MSE
            ##get the average and then sqrt an append
            #print(epoch_mse)
            overall_sqrt.append(np.sqrt(mean(epoch_mse)))
            ##get the average of r_squared for epoch and append
            overall_rsq.append(mean(epoch_rsq))
            ##calculate the inference latency fo each data set 
            overall_inference_latency.append(mean(pred_time))

        end_time = time.time()
        ##this is seconds
        total_time = end_time -start_time
        me_sqrt = mean(overall_sqrt)
        me_rsq = mean(overall_rsq)
        me_inference = mean(overall_inference_latency)

        columns = np.array([[str(self.configfile), str(self.model.__class__.__name__),loadertype,me_sqrt,total_time,me_inference,me_rsq]])        
        self.select_model = np.append(self.select_model, columns.transpose(),axis=1)
        
        ##this gets converted to a metric.json 
        self.model_metrics[loadertype]  = {
            'sqrt' : mean(overall_sqrt), 
            'r_squared' : mean(overall_rsq),
            'training_duration' : total_time,
            'inference_latency' : overall_inference_latency
            }                   

        
        time.sleep(1)


    def get_path(self,model,time):
            """
            Create directory for saving model results

            Arg:
                Takes model name
            
            Return:
                The pathway for storing  modellled data

            """

            modir = Path(f'models/regression/{model}/{time}')
            modir.mkdir(parents=True,exist_ok=True)

            return modir

    def get_nn_config(self, yfile =None):
        """
        Read in the neural network yaml config file
        Arg:
            Take file path to read

        Return:

            Dictionary of yaml contents
        """
        database_config = None
        ##note config stored at same level as this script
        with open(yfile) as file:
            try:
                database_config = yaml.safe_load(file)   
                print(database_config)
            except yaml.YAMLError as exc:
                print(exc)
        return database_config
    def generate_nn_config(self, lr,ly1=18):
        """
         Generates the config yaml file 

         Arg:

            lr:  Learning rate value for writing into yamil file
            ly1: The intial layer/depth 
        Return:

            Path for the new yaml config file 

        """
        #do it by factor of 9 based on the size  
        ly2 =ly1+10

        yaml_config = f"""        
            optimiser: torch.optim.SGD
            learning_rate: {lr}
            hidden_layer_width: 
                layer1: {ly1}
                layer2: {ly2}
                layer3: {1}
                depth:  10
        """
        modir = Path(f'models/config/{self.timestr()}')
        modir.mkdir(parents=True,exist_ok=True)
        yfile = yaml.safe_load(yaml_config)
        out = modir/ f"{ly1}_{lr}.yaml"
        with open(out,'w') as file:
            yaml.dump(yfile,file)

        return out

    def save_model(self):
        """
        Saves model into model directory
        Arg:
            
        Return
            Nothing
       """                   
        modir = self.get_path(str(self.model.__class__.__name__),self.timestr())
        torch.save(self.model.state_dict(), modir/ 'model.pt')
        ##write model details to file
        #joblib.dump(model,f"{modir}\\{nme}.joblib")
        ###jsonify the parameters
        
        jpath = modir/ "hyperparameters.json"
        ##numpyencoder deal with json.dumps limitations todealing with numpy arrays
        jpath.write_text(json.dumps(self.config,cls=NumpyEncoder))

        ##try data frame could use json.dumps
        hyperparam = pd.DataFrame(self.model_metrics)
        hyperparam.to_json(f"{modir}\metrics.json")

    def load(self,pt,ldtype):
        '''
        Redundant function
        '''

        state_dict = torch.load(pt)
        self.model = NN()
        self.model.load_state_dict(state_dict)
        self.train(loadertype=ldtype, epochs=10)

    def timestr(self):

        return (time.strftime("%Y-%m-%d_%H-%M-%S"))

class NN(torch.nn.Module):
    """
    Class which defines the neural network and layers width. Using linear regression with activation fucntion
    
    
    """
    def __init__(self,batchsize =2,config:dict = None) -> None:
        super().__init__()
        print(config)
        ##create and define layers. Use sequential as it saves setting up multiple steps for each layer.  This is a 3 layered Linear dip
        self.layers =  torch.nn.Sequential(
            torch.nn.Linear(batchsize,config['hidden_layer_width']['layer1']),
            torch.nn.Sigmoid(),
            torch.nn.Linear(config['hidden_layer_width']['layer1'],config['hidden_layer_width']['layer2']),
            torch.nn.Sigmoid(),
            torch.nn.Linear(config['hidden_layer_width']['layer2'],config['hidden_layer_width']['layer3'])
        )

    def forward(self,features):
        ##retrun prediction
        return self.layers(features)


class LinearRegression(torch.nn.Module):

    def __init__(self,batchsize =2) -> None:
        super().__init__()
        ##intiliase
        #batchsize = 9 # ???  << -- do not understand this if any other value will not work by states. 
        self.linear_layer = torch.nn.Linear(batchsize,1)

    def forward(self,features):
        ###need to cast these as float32 due to Linear unable to process float64
        #print(features)
               
        return self.linear_layer(features)



def evaluate_all_models(csv,labl):
    """
    Used to tes initial test set   
    
    """
    mgs = ModellingSearch()
    mgs.split_modelling(csv,labl)
    mgs.train(loadertype='train', epochs=9)
    mgs.train(loadertype='validation',epochs=9)
    mgs.train(loadertype='test',epochs=9)
    mgs.save_model()

def find_best_nn(csv,labl):  
    """
     Initialise and allows control the number of parameterisations  via multiples_numpy        
    
    """
    #muliples_numpy(width and iterations for optimisation)
    m9 = multiples_numpy(10,2)
    print("M( ==== ",m9)
    lr = [1,2,3,4,5]
    select_model = np.empty((7,0))
    
    for i in m9:
        for l in lr:
            ##get the list of config files  use learnign rate and layer size muliples of 9 due to with 9 columns per batch        
            configs = ModellingSearch().generate_nn_config(l,i)
            mgs = ModellingSearch(configs)
            mgs.split_modelling(csv,labl)            
            mgs.train(loadertype='train', epochs=9)
            mgs.train(loadertype='validation',epochs=9)
            mgs.train(loadertype='test',epochs=9)
            mgs.save_model()
            #with np.printoptions(threshold=np.inf):
                #print(mgs.select_model)
                ##ensure axis on otherwise flattens the arrays into 1D
            select_model = np.append(select_model, mgs.select_model, axis=1)

    mx_ac = max(select_model[-1])
    bst_hypeindex = np.where(select_model[-1]==mx_ac)[0][0]  #self.select_model[-1].index(mn_rmse_loss)
    # np.array([[str(self.configfile), str(self.model.__class__.__name__),loadertype,me_sqrt,total_time,me_inference,me_rsq]])
    model_selected = {
                'Best_model' : select_model[1][bst_hypeindex],
                'loadertype' : select_model[2][bst_hypeindex],
                'Config_file' : select_model[0][bst_hypeindex],
                'R2' : select_model[6][bst_hypeindex],
                'RMSE' : select_model[3][bst_hypeindex],
                'total_time' : select_model[4][bst_hypeindex],
                'Inference_latency' : select_model[5][bst_hypeindex]    
                
    }


    return (model_selected)

def multiples_numpy(value, length):
    return np.arange(1, length+1) * value



        

if __name__ == "__main__":

    csv = "tabular_data\listing.csv"
    labl = 'beds'

    #bst_mod = evaluate_all_models(csv,labl)
    bst_mod = find_best_nn(csv,labl)
    print(bst_mod)
   
    
    
