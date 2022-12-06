AirBNB

DataGrooming the AirbNB dataset containing following columns;

'ID	Category	Title	Description	Amenities	Location	guests	beds	bathrooms	Price_Night	Cleanliness_rating	Accuracy_rating	Communication_rating	Location_rating	Check-in_rating	Value_rating	amenities_count	url	bedrooms
'

Two classes created

1) Cleans the csv data
2) Process images 

Software:

1) tabular_data.py contains DataGroomer class plus selections of labels and building int64,float64 datatype features into panda datframe
2) prepare_image_data.py contains ImageProcessor class which  resize to a standard aspect ratio using min height as reference, from a set of images. cv is used for most of the processing

These two tools will allow for the application of machine learning algorithmns to be run over the cleaned data and processed images

#modelling.py

Software slices the data into validation, training and test setto predict price per night. The validation set is run against the 5 below regression models using GridSearchCV.
            ExtraTreesRegressor ,
            BaggingRegressor ,
            RandomForestRegressor,
            GradientBoostingRegressor ,
            DecisionTreeRegressor

Each model has relevant parmetes applied to them becuase nto all model have the naming conventions

The software stores the modelled data in to appropriately named model folder and parameters settings. 
The best model is then selected and then reported back.


History:

There is a method which allows custome tuning  are than the GridSearchCV wrapper class.   Cannot be accessed via command line but the way the ModellingGridSearch class is called will help.

