import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing


'''

This projects has a list of data from AirBNB but it needs tidying up first

1)There are missing values in the rating columns. Start by defining a function called remove_rows_with_missing_ratings which removes the rows with missing values in these columns. It should take in the dataset as a pandas dataframe and return the same type.

2) The "Description" column contains lists of strings. You'll need to define a function called combine_description_strings which combines the list items into the same string. Unfortunately, pandas doesn't recognise the values as lists, but as strings whose contents are valid Python lists. You should look up how to do this (don't implement a from-scratch solution to parse the string into a list). The lists contain many empty quotes which should be removed. If you don't remove them before joining the list elements with a whitespace, they might cause the result to contain multiple whitespaces in places. The function should take in the dataset as a pandas dataframe and return the same type. It should remove any records with a missing description, and also remove the "About this space" prefix which every description starts with.

3) The "guests", "beds", "bathrooms", and "bedrooms" columns have empty values for some rows. Don't remove them, instead, define a function called set_default_feature_values, and replace these entries with the number 1. It should take in the dataset as a pandas dataframe and return the same type.

Put all of the code that does this processing into a function called clean_tabular_data which takes in the raw dataframe, calls these functions sequentially on the output of the previous one, and returns the processed data.

Inside an if __name__ == "__main__" block, do the following things:

Load the raw data in using pandas
Call clean_tabular_data on it
Save the processed data as clean_tabular_data.csv in the same folder as you found the raw tabular data.
'''
class DataGroomer:

    def __init__(self,CSV):

        df = pd.read_csv(CSV, header=0)
        ##fill all empty cells with 1
        self.data = df.mask(df == '')


    def remove_rows_with_missing_ratings(self,column:list=None):

        self.data.dropna(subset=column,how='all',inplace=True)



    def combine_description_strings(self,column=None):

        ##change string to a list type
        self.data[column] = self.data[column].tolist() 
        ##rstart remove spurious quotes and sq brackets        
        self.data[column] = self.data[column].str.replace('\'','',regex=True).str.replace('\]','',regex=True).str.replace('""','"',regex=True)
        self.data.dropna(subset=column, inplace=True)      
        ##convert to a list 
        self.data[column] = self.data[column].str.split(',')
        #reconvert back to string now that it has mainly been tidied
        self.data[column] = self.data[column].str.join("")
        #last bit of tidying up
        self.data[column] = self.data[column].str.replace('\[About this space ','',regex=True).str.replace('"','',1)
       
        


    def set_default_feature_values(self,columns:list=None):

        ###make sure empty/NAN have 1 as a default value
        self.data[columns]=  self.data[columns].fillna(1)


def clean_tabular_data(CSV):

    dg = DataGroomer(CSV)
    dg.remove_rows_with_missing_ratings(['Cleanliness_rating','Accuracy_rating','Communication_rating','Location_rating','Check-in_rating','Value_rating'])
    dg.combine_description_strings('Description')
    dg.set_default_feature_values(["guests", "beds", "bathrooms", "bedrooms" ])
    ##gets rid of unamed colum which apprears from reading csv and writing to func
    dg.data.drop(dg.data.filter(regex="Unnamed"),axis=1, inplace=True)
    ##write CSV
    dg.data.to_csv("clean_data.csv",index=False)

    return dg

def load_airbnb(csv,labl=None,learning:str = None):

    
    dg = clean_tabular_data(csv)
    if  'classification' == learning.lower() :  
                
        features = dg.data.astype(str)      
        features.drop(['ID'],axis=1,inplace=True) 
    elif 'deep' == learning.lower():     
        features = dg.data
        le = preprocessing.LabelEncoder()
        features['Category'] = le.fit_transform(features['Category'])
        features.drop(['ID'],axis=1,inplace=True)   
        features = features.select_dtypes(['float64', 'int64','int32'])
    else:
        features = dg.data.select_dtypes(['float64', 'int64'])
    #print(features)
    df_features= pd.DataFrame(features)   

    ##ensure you use axis =1 for columns header otherwise tries rows
    ##ID column skews data.It provides no relevant information to the learning process 
    
    labels = df_features[labl]
    df_features.drop([labl],axis=1,inplace=True)
    print(df_features.dtypes)
    #print(df_features)
    return (df_features,labels)


class AirbnbNightlyPriceImageDataset(DataGroomer,Dataset):

    def __init__(self,csv,labl,learning):
        super().__init__(csv)
        self.features , self.labels = load_airbnb(csv,labl,learning)  
        
        #self.features = self.features.astype(int)    

    def __getitem__(self,idx):
        
        return(torch.tensor(self.features.iloc[idx]), self.labels.iloc[idx])

    def __len__(self):
        return len(self.features)
   

if __name__ == "__main__":

    csv = "tabular_data\listing.csv"
    labl = 'Price_Night'

    ###capture all the features and labels
    (features,labels) = load_airbnb(csv,labl)

