.. -*- mode: rst -*-
Integrated-AUTO-ML
============

Integrated AUTO ML is a python script for machine learning built on top of
scikit-learn. 

The project was started in 2019 by David Chen. During his internship at a technology start-up, he found out that in real-time practice, there is no highly integrated ML pipeline to help data scientists speed up both the model training and model implementation stages although scikit-learn is already a very powerful tool. So data scientists have to repeatedly copy and pasting the same code they developed for previous projects. 

Module introduction
------------
Module I is the data cleaning module. It helps to speed up the data cleaning process. 
Suppose a dataset has 1000 features, then data scientists have to go through all the features and decide what is the best actions one by one. But with the help of this tool, data scientists now have a dashboard to loop through all the features. During the looping process, some key statistics are revealed for that feature and the actions data scientists want to perform on the features will be recorded. So after the looping, data scientists can now process them all at once.

Module II is the feature checking module. It helps to reveal some statistics of the dataset before training.
With only a single line, data scientists can check which column has missing values and whether there are still features with object type in dataframe(which can not be processed by the scikit-learn models)

Module III is the auto ML module. It helps to perform k-fold cross-validation to find the best hyperparameters for the models.
It also helps to record all the potential information (such as what column is used, what is the final model, the necessary statistics needed for filling of missing value at new test set, etc.) that are needed during the model implementation process with only a few single lines during the final model building on all the dataset.
Before this tool, the work of the data scientist is tedious. Suppose I have a highly class-imbalanced dataset and have column A, B missing. Then during k-fold, I have to apply smote method on each train_fold and use the statistic such as the mean of the train_fold to fill the missing value on the validation_fold. The process is extremely tedious but now with this tool, only one single line of code can do that. What's more, during model implementation, what if the new test set have column C missing? Then data scientists have to open up the previous dataset and compute some statistic again(such as mean or max). But this module can help you save all the key information and perform all necessary transformation on the future test set.

Module IV is the auto Prediction module. It helps to apply what has been saved during the final training process(such as what column is used, what is the final model, the necessary statistics needed for filling of missing value at new test set, etc.) and automatically make the final prediction.


Installation
------------

Dependencies
~~~~~~~~~~~~

Integrated-AUTO-ML requires:
- scikit-learn (>= 0.20)
- matplotlib (>= 3.0)
- imblearn (>= 0.4)

Development
-----------

Demo_Titanic.ipynb is the Demo using Titanic dataset.
