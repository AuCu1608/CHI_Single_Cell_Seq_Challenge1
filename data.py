#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:31:02 2021

@author: CHI1
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
#import bcolz
#import pickle
#from nltk.tokenize import word_tokenize

'''
class DemoDataset(Dataset):
    def __init__(self, X, Y, emd_input_dim = 50,  col_name_emb_dict = None):        
        self.X = X.copy().values.astype(np.float32) # numetrical
        if col_name_emb_dict is None:
            rng = np.random.default_rng()
            self.X_col_cat_emb = rng.random((X.shape[1],emd_input_dim),dtype = np.float32)
        else:
            self.X_col_cat_emb = col_name_emb_dict.astype(np.float32)
            #self.X_col_cat_emb = np.random.random_sample((X.shape[1],50))
        #self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32) #numerical columns
        self.y = Y.astype(np.int64)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return  self.X_col_cat_emb, self.X[idx], self.y[idx]
'''   
    
def get_dataset(data_dir, return_X=True):
    classes = ['Normal', 'Mild', 'Severe']
    y = []
    labels = pd.read_csv(data_dir + "/covid-selected-data-labels.csv").set_index('Unnamed: 0').to_numpy()[:,0]
    for label in labels:
        y.append(classes.index(label))
        
    if not return_X:
        return np.array(y, dtype=int)
    
    dX = pd.read_csv(data_dir+"/covid-selected-data.csv").set_index('Unnamed: 0')
    
    return dX.to_numpy(), np.array(y, dtype=int),dX.columns.tolist()

def align_embfeatures(key_list, gene_emb_df):
    emd_array =  np.array([])
    for gene in key_list:
        emd_array = np.concatenate((emd_array, gene_emb_df[gene_emb_df['feat_name'] == gene].emb))
    
    return emd_array

class scBALFDataset(Dataset):
    def __init__(self, X, Y, emd_input_dim = 50,  col_name_emb_array = None):        
        self.X = X.copy().astype(np.float32) # numetrical
        if col_name_emb_array is None:
            rng = np.random.default_rng()
            self.X_col_cat_emb = rng.random((X.shape[1],emd_input_dim),dtype = np.float32)
        else:            
            self.X_col_cat_emb = col_name_emb_array.astype(np.float32)
            #self.X_col_cat_emb = np.random.random_sample((X.shape[1],50))
        #self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32) #numerical columns
        self.y = Y.astype(np.int64)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return  self.X_col_cat_emb, self.X[idx], self.y[idx]