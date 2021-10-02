# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import torch
# import seaborn as sns
from torch.utils.data import Dataset
import torch.optim as torch_optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
# import matplotlib.pyplot as plt    
import pickle
    
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def get_optimizer(model, lr = 0.001, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim


# def plot_heatmap(res, model_name, save_dir):
#     print('Model:', model_name)
#     classes = ['Normal', 'Mild', 'Severe']
#     y_pred, y_pred_proba, y_test = res['pre'].astype(int), res['pre_proba'], res['tru'].astype(int)
    
#     auroc_ovr_macro = roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovr')
#     auroc_ovr_weighted = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
#     auroc_ovo_macro = roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovo')
#     auroc_ovo_weighted = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovo')
#     print('AUROC OVR Macro: %.2f%%' % (auroc_ovr_macro*100))
#     print('AUROC OVR Weighted: %.2f%%' % (auroc_ovr_weighted*100))
#     print('AUROC OVO Macro: %.2f%%' % (auroc_ovo_macro*100))
#     print('AUROC OVO Macro: %.2f%%' % (auroc_ovo_weighted*100))
    
#     vis = confusion_matrix(y_test, y_pred)
#     accu = accuracy_score(y_test, y_pred)
    
#     #print(classification_report(y_test, y_pred))
#     print('Accuracy: %.2f%%' % (accu*100))
    
#     pd_vis = pd.DataFrame(vis, columns=classes, index=classes)
    
#     ax = sns.heatmap(pd_vis, cmap="Blues", annot=True, fmt='d')
    
#     #pd_vis.style.background_gradient(cmap = 'viridis')\
#     #            .set_properties(**{'font-size':'20px'})
    
#     # saving the figure.
#     plt.savefig(save_dir+"/confusion_matrix.png",
#                 bbox_inches ="tight",
#                 pad_inches = 1,
#                 transparent = False,
#                 facecolor ="w",
#                 edgecolor ='w',
#                 orientation ='landscape')
#     #plt.show()
#     result_dict = {'auroc_ovr_macro':auroc_ovr_macro,'auroc_ovr_weighted': auroc_ovr_weighted,
#                    'auroc_ovo_macro': auroc_ovo_macro,'auroc_ovo_weighted':auroc_ovo_weighted}
#     results_df = pd.DataFrame(result_dict, index = [0])
#     results_df.to_csv(save_dir+'/AUROC_results.csv')
#     return result_dict
    
def save_obj(obj,name):
    with open(name+'.pkl','wb') as f:
        pickle.dump(obj,f, pickle.HIGHEST_PROTOCOL)
        

def load_obj(name):
    with open(name+'.pkl','rb') as f:
        return pickle.load(f)