# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 07:24:00 2021

@author: Emma
"""

#import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from model import *
from utils import *
from data import *
import os
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd
# import pdb


def train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    weight_CE.to(device)
    for k, x, y in train_dl:
        k, x, y = k.to(device), x.to(device), y.to(device)
        batch = y.shape[0]
        out, attn = model(x, k)
        loss = F.cross_entropy(out, y, weight= weight_CE)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch*(loss.item())
    return sum_loss/total, attn

def val_loss(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0  
    attentions = []
    for k, x, y in valid_dl:
        current_batch_size = y.shape[0]
        out, attn = model(x, k)
        loss = F.cross_entropy(out, y, weight= weight_CE) 
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        correct += (pred == y).float().sum().item()
        attentions.append(attn)
    print("valid loss %.3f and accuracy %.3f" % (sum_loss/total, correct/total))
    return sum_loss/total, correct/total, attentions

def train_loop(model, epochs,  lr=0.01, wd=0.0, loop_id=0):
    optim = get_optimizer(model, lr = lr, wd = wd)
    best_accuracy_in_loop = 0
    for i in range(epochs): 
        loss, _ = train_model(model, optim, train_dl)
        print("Epoch ", i,": training loss: ", loss)
        _, val_accuracy, _ = val_loss(model, valid_dl)
        '''
        if epochs > warmup_epochs and val_accuracy > best_accuracy_in_loop:
            if model.mha:
                torch.save(model.state_dict(), path+subpath+'/best_model_mha'+ str(loop_id) +'.pt')
            else:                
                torch.save(model.state_dict(), path+subpath+'/best_model_'+ str(loop_id) +'.pt')
                best_accuracy_in_loop = val_accuracy  
        '''
    return best_accuracy_in_loop

def test_function(model,test_dl):
    preds = []
    pred_probas =  []
    correct = 0 
    total = 0
    with torch.no_grad():
        for k,x,y in test_dl:
            current_batch_size = y.shape[0]
            out, _ = model(x, k)
            prob = F.softmax(out, dim=1)
            pred_probas.append(prob)
            pred = torch.max(out, 1)[1]
            preds.append(pred)
            correct += (pred == y).float().sum().item()
            total += current_batch_size
        print("Test accuracy %.3f" % ( correct/total))
        #return pred_probas, preds
    preds = torch.cat(preds,dim=0)
    pred_probas = torch.cat(pred_probas,dim=0)
    return preds, pred_probas, correct/total



# data files: 
# 1.  "covid-selected-data-labels.csv" --- raw data
# 2.  "covid-selected-data.csv"  --- raw labels
# 3.  "idx_split.npy" --- fixed data split for duplication



# creat data and result path
path = 'test_results'    
if not os.path.exists(path):
    os.mkdir(path)

model_name = ['x_flow_deep_model', 'x_flow_shallow_model', 'x_flow_shallow_cat_emb', 'x_flow_deep_cat_emb']
datapath = "./scBALFdata"
    
for _name in model_name:     
    subpath = '/results_'+ _name            
    if not os.path.exists(path+subpath):
        os.makedirs(path+subpath)
    # hyperparameters of scBALF model
    input_dim = 1999
    emd_input_dim = 50
    hidden_dim_x_0 = 1000
    hidden_dim_x_1 = 200
    hidden_dim_cat = 30
    #multiheadattn = False  # the number of head is fix as 5 for simplicity
    
    # parameters for training the model
    batch_size = 16
    n_epochs = 20
    lr = 0.0004
    weight_decay = 0.00001
    n_splits_innercv = 5
    #warmup_epochs = 5    
    
    config = {'input_dim ': input_dim,
              'emd_input_dim ': emd_input_dim,
              'hidden_dim_x_0 ': hidden_dim_x_0,
              'hidden_dim_x_1 ': hidden_dim_x_1,
              'hidden_dim_cat': hidden_dim_cat,
              'batch_size ': batch_size,
              'n_epochs ': n_epochs,
              'lr' : lr,
              'weight_decay ': weight_decay,
              'n_splits_innercv ': n_splits_innercv,
              'dropout_rate':0.1
                }
    
    print('save experiment config...')
    save_obj(config,path + subpath + '/experiment_config')
    
    # load data
    print('loading raw data...')
    
    X, y, key_list = get_dataset(data_dir=datapath) 
    
    # load data split
    print('loading data split...')
    split_indices = np.load(datapath + '/idx_split.npy', allow_pickle=True).tolist()
    
    if os.path.isfile(datapath+'/gene_emb.npy'):
        gene_emb_array = np.load(datapath + '/gene_emb.npy')
    else:
        print('Gene embedding array is missing...')
    
    #model = scBALFModel(input_dim, emd_input_dim, hidden_dim, mha= multiheadattn)
    
    # push the model to device
    #device = get_default_device()
    #torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    y_pre, y_pre_proba, y_tru = [], [], np.array([])
    best_fold_auc = []
    weight_CE = torch.FloatTensor([3.5, 1, 2.5])
    weight_CE = weight_CE.to(device)
    
    classes = ['Normal', 'Mild', 'Severe']
    result_dict = defaultdict(list)
    fina_result_dict = defaultdict(dict)
    #test_aurocs = []
    test_aurocs = []
    for n_fold, (train_indices, test_indices) in enumerate(split_indices):        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]        
        
        skf = StratifiedKFold(n_splits=n_splits_innercv, shuffle=True, random_state=42)
        best_accuracy = 0
        # prepare test data
        test_ds = scBALFDataset(X_test, y_test, col_name_emb_array = gene_emb_array)
        test_dl = DataLoader(test_ds, batch_size=batch_size)
        test_dl = DeviceDataLoader(test_dl, device)    
        #inner_loop_index = 0
        for inner_n_fold, (inner_train_index, val_index) in enumerate(skf.split(X_train, y_train)):            
            print('Outter CV fold', n_fold, '---Inner CV fold', inner_n_fold, 6*'-')        
            X_inner_train, X_val = X_train[inner_train_index], X_train[val_index]
            y_inner_train, y_val = y_train[inner_train_index], y_train[val_index]          
    
            train_ds = scBALFDataset(X_inner_train, y_inner_train, col_name_emb_array = gene_emb_array)
            valid_ds = scBALFDataset(X_val, y_val, col_name_emb_array = gene_emb_array)
    
            train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
            valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)
    
            train_dl = DeviceDataLoader(train_dl, device)
            valid_dl = DeviceDataLoader(valid_dl, device)
            #print('pushing model to cuda...')  
            #model_name = ['x_flow_deep_model', 'x_flow_shallow_model', 'x_flow_vanilla_and_crossattn', 'x_flow_deel_and_crossattn']
            print('test model', _name)
            if _name is 'x_flow_deep_model':
                model = scBALFModel(d_x = input_dim, d_k = emd_input_dim, 
                                    hid_dim_x_0 = hidden_dim_x_0, hid_dim_x_1 = hidden_dim_x_1,
                                    hid_dim_cat= hidden_dim_cat, mask_k= 1)
            elif _name is 'x_flow_deep_cat_emb':
                model = scBALFModel(d_x = input_dim, d_k = emd_input_dim, 
                                    hid_dim_x_0 = hidden_dim_x_0, hid_dim_x_1 = hidden_dim_x_1,
                                    hid_dim_cat= hidden_dim_cat, mask_k= 0)
            elif _name is 'x_flow_shallow_model':
                 model = scBALFModel_shallow(d_x = input_dim, d_k = emd_input_dim, mask_k= 1)
            elif _name is 'x_flow_shallow_cat_emb':
                 model = scBALFModel_shallow(d_x = input_dim, d_k = emd_input_dim, mask_k= 0)
                 
                    
            model = model.to(device)
            # train and validate model in inner CV         
            best_acc_per_loop = train_loop(model, epochs=n_epochs, lr=lr, wd=weight_decay, loop_id = inner_n_fold)
            print('Best acc per loop:', best_acc_per_loop)
            print('Best accuracy:', best_accuracy)
            '''
            if best_acc_per_loop > best_accuracy:
                best_accuracy = best_acc_per_loop
                best_inner_loop_index = inner_n_fold
                print('Inner CV fold', inner_n_fold, ': best accuracy:', best_accuracy)
            else:
                print('performance is lower than best_acc, model is not saved!')
                #best_inner_loop_index = None
            '''
            
            y_test_pre, y_test_pre_proba, test_acc = test_function(model,test_dl) 
            auroc_ovr_macro = roc_auc_score(y_test_pre.cpu(), y_test_pre_proba.cpu(), average='macro', multi_class='ovr')
            print('TEST: AUROC OVR Macro: %.2f%%' % (auroc_ovr_macro*100))
            result_dict[n_fold].append(auroc_ovr_macro)
            torch.save(model.state_dict(), path+subpath+'/best_model_inner'+ str(inner_n_fold) +'.pt')
        
        # load the best model saved druing inner CV
        best_model_id_innerCV = np.argmax(result_dict[n_fold])
        
        
        if _name is 'x_flow_deep_model':
            best_model = scBALFModel(d_x = input_dim, d_k = emd_input_dim, 
                                hid_dim_x_0 = hidden_dim_x_0, hid_dim_x_1 = hidden_dim_x_1,
                                hid_dim_cat= hidden_dim_cat, mask_k= 1)
        elif _name is 'x_flow_deep_cat_emb':
            best_model = scBALFModel(d_x = input_dim, d_k = emd_input_dim, 
                                hid_dim_x_0 = hidden_dim_x_0, hid_dim_x_1 = hidden_dim_x_1,
                                hid_dim_cat= hidden_dim_cat, mask_k= 0)
        elif _name is 'x_flow_shallow_model':
             best_model = scBALFModel_shallow(d_x = input_dim, d_k = emd_input_dim, mask_k= 1)
        elif _name is 'x_flow_shallow_cat_emb':
             best_model = scBALFModel_shallow(d_x = input_dim, d_k = emd_input_dim, mask_k= 0)
        #best_model = scBALFModel(d_x = input_dim, d_k = emd_input_dim, 
        #                        hid_dim_x_0 = hidden_dim_x_0, hid_dim_x_1 = hidden_dim_x_1,
        #                        hid_dim_cat= hidden_dim_cat, mask_k= 0)
        best_model.load_state_dict(torch.load(path + subpath+'/best_model_inner'+ str(best_model_id_innerCV) +'.pt'))
        # make prediction
        best_model = best_model.to(device)
        y_test, y_pred_proba, test_acc = test_function(best_model,test_dl)
        # report and save test results 
        auroc_ovr_macro = roc_auc_score(y_test.cpu(), y_pred_proba.cpu(), average='macro', multi_class='ovr')
        auroc_ovr_weighted = roc_auc_score(y_test.cpu(), y_pred_proba.cpu(), average='weighted', multi_class='ovr')
        auroc_ovo_macro = roc_auc_score(y_test.cpu(), y_pred_proba.cpu(), average='macro', multi_class='ovo')
        auroc_ovo_weighted = roc_auc_score(y_test.cpu(), y_pred_proba.cpu(), average='weighted', multi_class='ovo')
        print('Test report--------')
        print('AUROC OVR Macro: %.2f%%' % (auroc_ovr_macro*100))
        print('AUROC OVR Weighted: %.2f%%' % (auroc_ovr_weighted*100))
        print('AUROC OVO Macro: %.2f%%' % (auroc_ovo_macro*100))
        print('AUROC OVO Macro: %.2f%%' % (auroc_ovo_weighted*100))        
        fina_result_dict['auroc_ovr_macro'][n_fold] = auroc_ovr_macro
        fina_result_dict['auroc_ovr_weighted'][n_fold] = auroc_ovr_weighted
        fina_result_dict['auroc_ovo_macro'][n_fold] = auroc_ovo_macro
        fina_result_dict['auroc_ovo_weighted'][n_fold] = auroc_ovo_weighted    
        df = pd.DataFrame(fina_result_dict.items(), columns=['name', 'fold:value'])
        avar = [np.mean(list(fina_result_dict['auroc_ovr_macro'].values())),
                np.mean(list(fina_result_dict['auroc_ovr_weighted'].values())),
                np.mean(list(fina_result_dict['auroc_ovo_macro'].values())),
                np.mean(list(fina_result_dict['auroc_ovo_weighted'].values()))
                ]
        std = [np.std(list(fina_result_dict['auroc_ovr_macro'].values())),
               np.std(list(fina_result_dict['auroc_ovr_weighted'].values())),
               np.std(list(fina_result_dict['auroc_ovo_macro'].values())),
               np.std(list(fina_result_dict['auroc_ovo_weighted'].values()))
               ]
        
        df['mean'] =avar
        df['std'] = std
        #save results and model
        df.to_csv(path+subpath+'/aurocs.csv')
        torch.save(best_model.state_dict(), path+subpath+'/best_model_outter'+ str(n_fold) +'.pt')
        del best_model
        
        print('Outter CV fold', n_fold, ": best test auc %.3f " % (auroc_ovr_macro))
    
