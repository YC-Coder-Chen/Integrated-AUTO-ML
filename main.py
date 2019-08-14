#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:10:21 2019

@author: chenyingxiang
"""
#%%
import os
import pickle
import pandas as pd
import numpy as np
from IPython.display import clear_output
import itertools
import warnings
warnings.filterwarnings("ignore")
import operator
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn import model_selection as sk_ms
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif

class data_cleaning:
    def __init__(self, df, candi_col, tar_col, actions_dict):
        self.df = df
        self.candi_col = sorted(candi_col)
        self.tar_col = tar_col
        self.actions_dict = actions_dict
    
    def update_candi_col(self, candi_col):
        self.candi_col = sorted(candi_col)
    
    def update_actions_dict(self, actions_dict):
        self.actions_dict = actions_dict

    def main(self, idx=0):
        df = self.df.copy()
        list_idx = list(range(len(self.candi_col)))
        actions_dict = self.actions_dict
        action = dict()
        clear_output()
        
        def inner_loop(idx, candi_col, list_idx):
            if idx<len(candi_col):
                for num in list_idx[idx:]:  
                    col = self.candi_col[num]
                    print('Columns: ', col)
                    print('Columns type: ', df[col].dtypes)
                    print('\n')
                    try:
                        print('# of Uniques: ' + str(len(df[col].unique())))
                        print('\n')
                    except:
                        print('Cannot hash the data in the column')
                    print('Sample of the column: \n')
                    print(df[col].sample(10))
                    print('\n')
                    print('########################################')
                    print(actions_dict)
                    print('\n')
                    print('########################################')
                    try:
                        ne1 = self.candi_col[num + 1]
                        print('NEXT1:    ',ne1, df[ne1].dtypes)
                    except:
                        None
                    try:
                        ne2 = self.candi_col[num + 2]
                        print('NEXT2:    ',ne2, df[ne2].dtypes)
                    except:
                        None
                    print('\n')
                    print('Progress: ' + str(num) + '/' + str(len(self.candi_col)))
                    print('\n')

                    input_result = input('Your choice')
                    if input_result!='break':
                        action[col] = input_result
                        clear_output()
                    else:
                        idx = input('where you want to start?')
                        clear_output()
                        inner_loop(int(idx),candi_col, list_idx)
                        break
                        clear_output()
        inner_loop(idx, self.candi_col, list_idx) 
        return action 

class feature_check:
    def __init__(self, df, tar_col):
        self.df = df
        self.tar_col = tar_col
        
    def check_missing(self):
        df_missing = self.df.isnull().sum().reset_index()
        df_missing.columns = ['features','#null']
        return df_missing, list(df_missing[df_missing['#null']>0]['features'].values)
    
    def check_object(self):
        return list(self.df.select_dtypes(['object']).columns)==[]
    
    def check_imblance(self):
        return self.df[self.tar_col].value_counts()

class autoML:
    def __init__(self, df, tar_col):
        self.df = df
        self.tar_col = tar_col
        data_dir="./data/"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
    
    def split(self, size = 0.2, train=[], test=[]):
        if train!=[] and test==[]:
            raise ValueError('Pls provide both train and test')
        if train==[] and test!=[]:
            raise ValueError('Pls provide both train and test')        
        if train!=[] and test!=[]:
            self.train = train
            self.test = test
        if train==[] and test==[]:
            self.train, self.test = sk_ms.train_test_split(self.df, test_size=size)
        
    def cross_validation(self, fold, col, model, params, scorer, fill_method = False, scaling = False, 
                         fkbest=None, smote=False, classification=True):
        train = self.train.copy()
        target = self.tar_col
        result_dict = dict() 
        kf = KFold(n_splits=fold)
        params_list = list(itertools.product(*params.values()))
        if fkbest==None:
            fkbest = [len(col)]
        for best_n in fkbest:
            for param in params_list:
                param_dict = dict(zip(params.keys(),param)) 
                res = []
                for train_index, test_index in kf.split(train):
                    fold_train = train.iloc[train_index,:].copy()
                    fold_valid = train.iloc[test_index,:].copy()
                    
                    if (fold_train[list(set(col) - set([target]))].isnull().sum().sum()!=0) and (fill_method == False):
                        raise ValueError('missing value found in the dataset')
                    if (fold_valid[list(set(col) - set([target]))].isnull().sum().sum()!=0) and (fill_method == False):
                        raise ValueError('missing value found in the dataset')

                    if fill_method == 'mean':
                        for column in list(set(col) - set([target])):
                            mean = fold_train[column].mean()
                            fold_train.loc[:,column] = fold_train.loc[:,column].fillna(mean).copy()
                            fold_valid.loc[:,column] = fold_valid.loc[:,column].fillna(mean).copy() 
                    
                    if fill_method == 'zero':
                        for column in list(set(col) - set([target])):
                            fold_train.loc[:,column] = fold_train.loc[:,column].fillna(0).copy()
                            fold_valid.loc[:,column] = fold_valid.loc[:,column].fillna(0).copy() 
                    
                    if fill_method == 'max':
                        for column in list(set(col) - set([target])):
                            max_value = fold_train[column].max()
                            fold_train.loc[:,column] = fold_train.loc[:,column].fillna(max_value).copy()
                            fold_valid.loc[:,column] = fold_valid.loc[:,column].fillna(max_value).copy() 
                    
                    if fill_method == 'min':
                        for column in list(set(col) - set([target])):
                            min_value = fold_train[column].min()
                            fold_train.loc[:,column] = fold_train.loc[:,column].fillna(min_value).copy()
                            fold_valid.loc[:,column] = fold_valid.loc[:,column].fillna(min_value).copy()                     
                    
                    if fill_method == 'median':
                        for column in list(set(col) - set([target])):
                            median_value = fold_train[column].median()
                            fold_train.loc[:,column] = fold_train.loc[:,column].fillna(median_value).copy()
                            fold_valid.loc[:,column] = fold_valid.loc[:,column].fillna(median_value).copy()
                    
                    if fill_method == 'mode':
                        for column in list(set(col) - set([target])):
                            mode_value = fold_train[column].mode().values[0]
                            fold_train.loc[:,column] = fold_train.loc[:,column].fillna(mode_value).copy()
                            fold_valid.loc[:,column] = fold_valid.loc[:,column].fillna(mode_value).copy() 
                            
                    if isinstance(fill_method, dict):
                        for column in list(set(col) - set([target])):
                            if column in fill_method.keys():
                                value = fill_method.get(column)
                                fold_train.loc[:,column] = fold_train.loc[:,column].fillna(value).copy()
                                fold_valid.loc[:,column] = fold_valid.loc[:,column].fillna(value).copy() 

                    if (fold_train[list(set(col) - set([target]))].isnull().sum().sum()!=0):
                        raise ValueError('missing value found in the dataset')
                    if (fold_valid[list(set(col) - set([target]))].isnull().sum().sum()!=0):
                        raise ValueError('missing value found in the dataset')                               

                    if scaling:
                        for column in list(set(col) - set([target])):
                            mean = fold_train[column].mean()
                            std = fold_train[column].std()
                            max_value = fold_train.loc[:,column].max()
                            min_value = fold_train.loc[:,column].min() 
                            
                            train_value = ((fold_train.loc[:,column] - mean)/std)\
                                    .replace([np.inf], max_value).replace([-np.inf], min_value)
                            
                            valid_value = ((fold_valid.loc[:,column] - mean)/std)\
                                    .replace([np.inf], max_value).replace([-np.inf], min_value)
                            train_value = train_value.fillna(train_value.mean()).fillna(0)
                            valid_value = valid_value.fillna(train_value.mean()).fillna(0)
                            fold_train.loc[:,column] = train_value
                            fold_valid.loc[:,column] = valid_value
                            
                    fold_train_x = fold_train[list(set(col) - set([target]))].copy()
                    fold_train_y = fold_train[target].copy() 

                    if fkbest==[len(col)]:
                        fvalue_selector = SelectKBest(f_classif, k=best_n)
                        fvalue_selector.fit_transform(fold_train_x, fold_train_y)
                        feature_list = fold_train_x.columns[fvalue_selector.get_support(indices=True)].tolist()
                        fold_train_x = fold_train[feature_list].copy()
                        fold_train_y = fold_train[target].copy()

                    if classification and smote:
                        sm = SMOTE()
                        col_smote = fold_train_x.columns
                        fold_train_x, fold_train_y = sm.fit_sample(fold_train_x,
                                                                   fold_train_y)
                        fold_train_x = pd.DataFrame(fold_train_x, 
                                                    columns = col_smote).copy()
                    col_valid = fold_train_x.columns
                    fold_valid_x = fold_valid[col_valid].copy()
                    fold_valid_y = fold_valid[target].copy()

                    estimator = model(**param_dict)
                    tree = estimator.fit(X = fold_train_x, y = fold_train_y.ravel())
                    try:
                        prediction = tree.predict_proba(fold_valid_x)[:,1]
                    except:
                        prediction = tree.predict(fold_valid_x)
                    res.append(scorer(fold_valid_y.values, prediction))
                result_dict[tuple(list(param)+[best_n])] = np.array(res).mean()
        
        return (result_dict, max(result_dict.items(), key=operator.itemgetter(1))
                ,min(result_dict.items(), key=operator.itemgetter(1)))
    
    def predict_test(self, col, model, best_params, scorer, fill_method = False, scaling = False, 
                         fkbest=None, smote=False, classification=True):
        train = self.train.copy()
        evaluation = self.test.copy()
        target = self.tar_col
        params_list = list(itertools.product(*best_params.values()))
        
        if fkbest==None:
            fkbest = [len(col)]        
        best_n = fkbest
        param_dict = dict(zip(best_params.keys(),params_list[0])) 
        
        
        if (train[list(set(col) - set([target]))].isnull().sum().sum()!=0) and (fill_method == False):
            raise ValueError('missing value found in the dataset')
        if (evaluation[list(set(col) - set([target]))].isnull().sum().sum()!=0) and (fill_method == False):
            raise ValueError('missing value found in the dataset')

        if fill_method == 'mean':
            for column in list(set(col) - set([target])):
                mean = train[column].mean()
                train.loc[:,column] = train.loc[:,column].fillna(mean).copy()
                evaluation.loc[:,column] = evaluation.loc[:,column].fillna(mean).copy() 

        if fill_method == 'zero':
            for column in list(set(col) - set([target])):
                train.loc[:,column] = train.loc[:,column].fillna(0).copy()
                evaluation.loc[:,column] = evaluation.loc[:,column].fillna(0).copy() 

        if fill_method == 'max':
            for column in list(set(col) - set([target])):
                max_value = train[column].max()
                train.loc[:,column] = train.loc[:,column].fillna(max_value).copy()
                evaluation.loc[:,column] = evaluation.loc[:,column].fillna(max_value).copy() 

        if fill_method == 'min':
            for column in list(set(col) - set([target])):
                min_value = train[column].min()
                train.loc[:,column] = train.loc[:,column].fillna(min_value).copy()
                evaluation.loc[:,column] = evaluation.loc[:,column].fillna(min_value).copy()                     

        if fill_method == 'median':
            for column in list(set(col) - set([target])):
                median_value = train[column].median()
                train.loc[:,column] = train.loc[:,column].fillna(median_value).copy()
                evaluation.loc[:,column] = evaluation.loc[:,column].fillna(median_value).copy()

        if fill_method == 'mode':
            for column in list(set(col) - set([target])):
                mode_value = train[column].mode().values[0]
                train.loc[:,column] = train.loc[:,column].fillna(mode_value).copy()
                evaluation.loc[:,column] = evaluation.loc[:,column].fillna(mode_value).copy() 

        if isinstance(fill_method, dict):
            for column in list(set(col) - set([target])):
                if column in fill_method.keys():
                    value = fill_method.get(column)
                    train.loc[:,column] = train.loc[:,column].fillna(value).copy()
                    evaluation.loc[:,column] = evaluation.loc[:,column].fillna(value).copy() 

        if (train[list(set(col) - set([target]))].isnull().sum().sum()!=0):
            raise ValueError('missing value found in the dataset')
        if (evaluation[list(set(col) - set([target]))].isnull().sum().sum()!=0):
            raise ValueError('missing value found in the dataset') 

        if scaling:
            for column in list(set(col) - set([target])):
                mean = train[column].mean()
                std = train[column].std()
                max_value = train.loc[:,column].max()
                min_value = train.loc[:,column].min() 

                train_value = ((train.loc[:,column] - mean)/std)\
                        .replace([np.inf], max_value).replace([-np.inf], min_value)

                eval_value = ((evaluation.loc[:,column] - mean)/std)\
                        .replace([np.inf], max_value).replace([-np.inf], min_value)
                train_value = train_value.fillna(train_value.mean()).fillna(0)
                eval_value = eval_value.fillna(train_value.mean()).fillna(0)
                train.loc[:,column] = train_value
                evaluation.loc[:,column] = eval_value

        train_x = train[list(set(col) - set([target]))].copy()
        train_y = train[target].copy() 

        if fkbest==[len(col)]:
            fvalue_selector = SelectKBest(f_classif, k=best_n)
            fvalue_selector.fit_transform(train_x, train_y)
            feature_list = train_x.columns[fvalue_selector.get_support(indices=True)].tolist()
            train_x = train[feature_list].copy()
            train_y = train[target].copy()

        if classification and smote:
            sm = SMOTE()
            col_smote = train_x.columns
            train_x, train_y = sm.fit_sample(train_x,train_y)
            train_x = pd.DataFrame(train_x, columns = col_smote).copy()
            
        col_eval = train_x.columns
        evaluation_x = evaluation[col_eval].copy()
        evaluation_y = evaluation[target].copy()

        estimator = model(**param_dict)
        tree = estimator.fit(X = train_x, y = train_y.ravel())
        self.test_model = tree
        
        try:
            prediction = tree.predict_proba(evaluation_x)[:,1]
        except:
            prediction = tree.predict(evaluation_x)
        
        if classification==True:
            thres = np.array([0.05*i for i in range(20)])
            def precision_cal(x):
                tn, fp, fn, tp = confusion_matrix(evaluation_y, prediction>=x).ravel()
                return tp/(tp+fp)
            precision = [precision_cal(i) for i in thres]

            def recall_cal(x):
                tn, fp, fn, tp = confusion_matrix(evaluation_y, prediction>=x).ravel()
                return tp/(tp+fn)
            recall = np.array([recall_cal(i) for i in thres])
            
            line_up, = plt.plot(thres, precision, 'r--', label='Precision')
            line_down, = plt.plot(thres, recall, 'b--', label='Recall')
            plt.legend(handles=[line_up, line_down])
            plt.show()
            
        return (tree, prediction)
    
    def final_product(self, col, model, best_params, scorer, fill_method = False, scaling = False, 
                         fkbest=None, smote=False, classification=True):
        
        train = self.train.copy()
        evaluation = self.test.copy()
        all_data = train.append(evaluation)   
        target = self.tar_col
        params_list = list(itertools.product(*best_params.values()))  
        
        if fkbest==None:
            fkbest = [len(col)]        
        best_n = fkbest
        param_dict = dict(zip(best_params.keys(),params_list[0])) 
        
        if (all_data[list(set(col) - set([target]))].isnull().sum().sum()!=0) and (fill_method == False):
            raise ValueError('missing value found in the dataset')

        if fill_method == 'mean':
            for column in list(set(col) - set([target])):
                mean = all_data[column].mean()
                all_data.loc[:,column] = all_data.loc[:,column].fillna(mean).copy()

            all_data_fill = all_data[list(set(col) - set([target]))].mean()
            all_data_fill  = all_data_fill.to_dict()
            with open("./data/final_fill.pkl", "wb") as fp:  
                pickle.dump(all_data_fill, fp)                     

        if fill_method == 'zero':
            for column in list(set(col) - set([target])):
                all_data.loc[:,column] = all_data.loc[:,column].fillna(0).copy()

            temp = all_data[list(set(col) - set([target]))].mean()
            all_data_fill = pd.Series(0, index=temp.index)
            all_data_fill  = all_data_fill.to_dict()
            with open("./data/final_fill.pkl", "wb") as fp:  
                pickle.dump(all_data_fill, fp)   

        if fill_method == 'max':
            for column in list(set(col) - set([target])):
                max_value = all_data[column].max()
                all_data.loc[:,column] = all_data.loc[:,column].fillna(max_value).copy()

            all_data_fill = all_data[list(set(col) - set([target]))].max()
            all_data_fill  = all_data_fill.to_dict()
            with open("./data/final_fill.pkl", "wb") as fp:  
                pickle.dump(all_data_fill, fp) 

        if fill_method == 'min':
            for column in list(set(col) - set([target])):
                min_value = all_data[column].min()
                all_data.loc[:,column] = all_data.loc[:,column].fillna(min_value).copy()                    

            all_data_fill = all_data[list(set(col) - set([target]))].min()
            all_data_fill  = all_data_fill.to_dict()
            with open("./data/final_fill.pkl", "wb") as fp:  
                pickle.dump(all_data_fill, fp)                            

        if fill_method == 'median':
            for column in list(set(col) - set([target])):
                median_value = all_data[column].median()
                all_data.loc[:,column] = all_data.loc[:,column].fillna(median_value).copy()

            all_data_fill = all_data[list(set(col) - set([target]))].median()
            all_data_fill  = all_data_fill.to_dict()
            with open("./data/final_fill.pkl", "wb") as fp:  
                pickle.dump(all_data_fill, fp) 

        if fill_method == 'mode':
            for column in list(set(col) - set([target])):
                mode_value = all_data[column].mode().values[0]
                all_data.loc[:,column] = all_data.loc[:,column].fillna(mode_value).copy()

            all_data_fill = all_data[list(set(col) - set([target]))].mode().iloc[0]
            all_data_fill  = all_data_fill.to_dict()
            with open("./data/final_fill.pkl", "wb") as fp:  
                pickle.dump(all_data_fill, fp) 

        if isinstance(fill_method, dict):
            for column in list(set(col) - set([target])):
                if column in fill_method.keys():
                    value = fill_method.get(column)
                    all_data.loc[:,column] = all_data.loc[:,column].fillna(value).copy()

            with open("./data/final_fill.pkl", "wb") as fp:  
                pickle.dump(fill_method, fp) 

        if (all_data[list(set(col) - set([target]))].isnull().sum().sum()!=0):
            raise ValueError('missing value found in the dataset')
            
        
        if scaling:            
            # save all the means
            all_data_mean = all_data[list(set(col) - set([target]))].mean()
            all_data_mean  = all_data_mean.to_dict()
            with open("./data/final_mean.pkl", "wb") as fp:  
                pickle.dump(all_data_mean, fp)       
            
            # save all the stds
            all_data_std = all_data[list(set(col) - set([target]))].std()
            all_data_std  = all_data_std.to_dict()
            with open("./data/final_std.pkl", "wb") as fp:  
                pickle.dump(all_data_std, fp)   
            
            for column in list(set(col) - set([target])):
                mean = train[column].mean()
                std = train[column].std()
                max_value = train.loc[:,column].max()
                min_value = train.loc[:,column].min() 

                all_data_value = ((all_data.loc[:,col] - mean)/std)\
                        .replace([np.inf], max_value).replace([-np.inf], min_value)

                all_data_value = all_data.fillna(all_data_value.mean()).fillna(0)
                all_data.loc[:,column] = all_data_value
            
        all_data_x = all_data[list(set(col) - set([target]))].copy()
        all_data_y = all_data[target].copy()             
        
        if fkbest==[len(col)]:
            fvalue_selector = SelectKBest(f_classif, k=best_n)
            fvalue_selector.fit_transform(all_data_x, all_data_y)
            feature_list = all_data_x.columns[fvalue_selector.get_support(indices=True)].tolist()
            all_data_x = all_data[feature_list].copy()
            all_data_y = all_data[target].copy()

        if classification and smote:
            sm = SMOTE()
            col_smote = all_data_x.columns
            all_data_x, all_data_y = sm.fit_sample(all_data_x, all_data_y)
            all_data_x = pd.DataFrame(all_data_x, columns = col_smote).copy() 

        estimator = model(**param_dict)
        tree = estimator.fit(X = all_data_x, y = all_data_y.ravel())
        self.final_model = tree  
        
        with open("./data/final_model.pkl", "wb") as fp:  
                pickle.dump(tree, fp)
        
        with open("./data/final_params.pkl", "wb") as fp:  
                pickle.dump(best_params, fp)        
        
        final_feature = list(set(col) - set([target]))
        with open("./data/final_feature.pkl", "wb") as fp:  
            pickle.dump(final_feature, fp)
        
        print('Prediction Finished and saved!')
