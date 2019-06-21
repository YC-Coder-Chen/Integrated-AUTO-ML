import pandas as pd
import numpy as np
from IPython.display import clear_output

class ds_tools:
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
                    except:
                        print('Cannot hash the data in the column')
                    print(df[col].head(10))
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