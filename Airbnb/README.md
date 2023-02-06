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

Software slices the data into validation, training and test set to predict price per night. The validation set is run against the 5 below regression models using GridSearchCV.
            ExtraTreesRegressor ,
            BaggingRegressor ,
            RandomForestRegressor,
            GradientBoostingRegressor ,
            DecisionTreeRegressor

Each model has relevant parmetes applied to them because not all models have the same naming conventions

The software stores the modelled data in to appropriately named model folder and parameters settings. 
The best model is then selected and then reported back.


History:

There is a method which allows custom tuning  rather than use the GridSearchCV wrapper class.   Cannot be accessed via command line but the way the ModellingGridSearch class is called will allow access to it. See evaluate_all_models() more order of calling


#classification_modelling.py

This scripts is built on modelling.py but asseses the below classifers to determine the Category each room belongs in and their accuacy of predicting it. A full Grid search is used to determine the best parameters. T???talk about overfitting 

            LogisticRegression 
            RandomForestClassifier 
            KNeighborsClassifier 
            MLPClassifier 
            AdaBoostClassifier 

All modelling like previous software store the models

Comparison was made betweeen the accuracy of the training set vs test (15%) set.  1st pass demonstrated that overfitting is likely with the training set  achieving 100% accuracy while the test 32 %
Best_model : LogisticRegression
Parameters : 2.0_1000_l2_liblinear_True
Validation_accuracy : 0.45161290322580644
Accuracy_train : 1.0
Accuracy_test : 0.32  

To prevent this overfitting adjstment were made so more data was used for training.  The data split ratios were adjusted in favour of this with test set being 10% with training now at 80%

Best_model : KNeighborsClassifier
Parameters : euclidean_19
Validation_accuracy : 0.40963855421686746
Accuracy_train : 0.42168674698795183
Accuracy_test : 0.37349397590361444

This led to  KNeighbours having a better validation score with the Training and test set closer in accuracy. But this method selected a different model to when the data had 15% test. 

The process was repeated. 

Best_model : LogisticRegression
Parameters : 3.0_1000_l2_liblinear_True
Validation_accuracy : 0.39759036144578314
Accuracy_train : 1.0
Accuracy_test : 0.3855421686746988

Yet again Logistic regression performs better with the validation data. The optimal parameter has a modification of the regularisation  Still seems to have overfit

Process was run for 3rd time at test still 10% of data

Best_model : AdaBoostClassifier
Parameters : 1_16
Validation_accuracy : 0.43373493975903615
Accuracy_train : 0.4036144578313253
Accuracy_test : 0.3855421686746988

3 different models selected so far.  With a high learning rate set at 16.  Adaboost in default setting use DecisionTreeClassifier as default.

Boosting is known to be use ensembble methods to improve weak learner accuracy.  Looking at the model with the lowest variance between their training and test set I can see Logistic regression tends to overfit the data but also generally have a reasonably higher accuracy than the other model.   But it does overfit.  RandomforesClassifier seems to consistenly  rank second for accuracy.  It was worth exploring boosting with this bagging method.


Best_model : LogisticRegression
Parameters : 2.0_1000_l2_liblinear_True
Validation_accuracy : 0.4578313253012048
Accuracy_train : 1.0
Accuracy_test : 0.4939759036144578

Logistic regression still beats the accuracy against adaboost.  There is no noticeable difference in accuracy than using the default base estimator for adaboost.  There is no appreciable difference that running Randomforest by itself. Discussion of combining bagging methods with boosting doesn't seem to support combining it.  No link to change but Logistic regression has so far given the highest accuracy rate

I used logistic regression  as base estimator with adaboost to see if adaboost is able to compensate for overfitting while increasing accuracy. 

Best_model : LogisticRegression
Parameters : 3.0_1000_l2_liblinear_True
Validation_accuracy : 0.40963855421686746
Accuracy_train : 1.0
Accuracy_test : 0.37349397590361444

Despite adding logistic regression as a base estimator for Adaboost, looks like logistic regression performs best.

You can see that three of the hyperparameters stay constant 1000, l2, liblinear while the Regulairsation fluctuates between 2 and 3.  Therefore a new range can be provided for this 
[0.1,0.2,1,2,2.1, 2.2, 2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,4]
Best_model : LogisticRegression
Parameters : 1.0_1000_l1_liblinear_True
Validation_accuracy : 0.37349397590361444
Accuracy_train : 0.6897590361445783
Accuracy_test : 0.3855421686746988

For the first time the accuracy for training has decreased. But the Regularisation range change was between 2-3 yet 1 was selected.  
[0.1,0.2,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1, 2.2, 2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,4]


The point here is that more optimisaiton is possible. 

Note
There has consistently been an issue with Liblinear convergence yet the choice was 1000 or 5000 and 5000 has never been selected as the best parameters. Therefore selection was done. Therefore 1000 was removed and 1500 also added 'max_iter' : [1500,2000,2500,3000,3500,5000]

Liblinear convergence error were still noted even with min increased to 1500


#Deep Learning model

The deep_learning_modelling.py looks at the AIRBnb data 

It first builds a data loader  which shuffles the data and then splits it into 3 datasets validation,training and testing
A neural network class has ben built using pytorch linear, and a Sigmoid activation function to prevent oversimplification of model predictions.
The dataloader passes the data to the NN class and data is batched accordingly.  16 different parameterisation are selected and all done by factor of 9
Every parameterisation is store in the hyperparameters.json, metrics json and model.pt file.  The best model instance is selected and reported.

Model is able to deal with labels by hotencoding them. For the AiBNB project this category column has been labelled and converted to int32 value.

##possible future feature to be able to autmatically convert all labels but this is left to ensure you as a user understand the datatype of the features before fully making decision top convert dtypes.






