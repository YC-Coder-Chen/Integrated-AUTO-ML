#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:10:21 2019

@author: chenyingxiang
"""
#%%
import pandas as pd
import numpy as np
from IPython.display import clear_output

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

#%%
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
    
    def check_imblance(sedf):
        return self.df[tar_col].value_counts()

#%%
import itertools
import warnings
warnings.filterwarnings("ignore")
import operator
from imblearn.over_sampling import SMOTE
from sklearn import model_selection as sk_ms
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif

class autoML:
    def __init__(self, df, tar_col):
        self.df = df
        self.tar_col = tar_col
    
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
        
    def cross_validation(self, fold, col, model, params, scorer, fill_mean = False, scaling = False, 
                         fkbest=None, smote=False, classification=False):
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

                    if fill_mean:
                        for col in [list(set(col) - set([target]))]:
                            mean = fold_train[col].mean()
                            fold_train.loc[:,col] = fold_train.loc[:,col].fillna(mean).copy()
                            fold_valid.loc[:,col] = fold_valid.loc[:,col] .fillna(mean).copy() 

                    if scaling:
                        for col in [list(set(col) - set([target]))]:
                            mean = fold_train[col].mean()
                            std = fold_train[col].std()
                            train_value = (fold_train.loc[:,col] - mean)/std
                            valid_value = (fold_valid.loc[:,col] - mean)/std
                            train_value = train_value.fillna(train_value.max())
                            valid_value = valid_value.fillna(train_value.max()) 
                            fold_train.loc[:,col] = train_value
                            fold_valid.loc[:,col] = valid_value
                            
                    fold_train_x = fold_train[list(set(col) - set([target]))].copy()
                    fold_train_y = fold_train[target].copy() 

                    if fkbest==[len(col)]:
                        fvalue_selector = SelectKBest(f_classif, k=best_n)
                        x_temp = fvalue_selector.fit_transform(fold_train_x, fold_train_y)
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

#%%