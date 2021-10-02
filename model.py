# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 07:23:21 2021

@author: Emma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """ Cross-Attention """
    def __init__(self, d_x, d_k):
        super().__init__()            
        self.bn = nn.BatchNorm1d(d_x)
        self.lin = nn.Linear(d_x, d_k)        
        self.softmax = nn.Softmax(dim = 2)  
        
    def forward(self, x, k, mask=None):
        latent_x = self.bn(x)
        latent_x = self.lin(torch.unsqueeze(latent_x,1)) # 1. latent_x : batch *  1 * d_k
        u = torch.bmm(k, latent_x.transpose(1,2)) # 2. Matmal: batch * d_x * 1
        
        attn = self.softmax(u) # 3. Softmax : batch * d_x * 1
        output = torch.bmm(k.transpose(1,2), attn) # 4. Output: 
 
        return torch.squeeze(attn,2), torch.squeeze(output,2), torch.squeeze(latent_x.transpose(1,2),2)

class MultiHeadCrossAttention(nn.Module):
    """ Multi-Head Cross Attention """

    def __init__(self, n_head, d_x_, d_k_, d_k, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_x = d_k
        
        self.bn = nn.BatchNorm1d(d_x_)        
        self.fc_x = nn.Linear(d_x_, n_head * d_k)  # 
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.attention = CrossAttention(self.d_x, self.d_k)
        self.fc_o = nn.Linear(n_head * d_k, d_o)  # 

    def forward(self, x, k, mask=None):

        n_head, d_x, d_k, = self.n_head, self.d_x, self.d_k 
        
        x = self.bn(x) # x: (batch, d_x_) 
        x = torch.unsqueeze(x,1) # x: (batch, 1, d_x_)        
        
        batch, n_x, d_x_ = x.size()
        batch, n_k, d_k_ = k.size()          
        
        x = self.fc_x(x) # 1.  x (batch ,  1 , n_head * d_k) 

        k = self.fc_k(k) # k: (batch, d_x ,n_head * d_k)
        
        x = x.view(batch, n_x, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_x, d_k)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
       
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        
        x = torch.squeeze(x,1) # CrossAttention takes x (2D) as input
        attn, output, x_hid = self.attention(x, k, mask=mask) # 2.
       
        
        output = output.view(n_head, batch, n_x, d_k).permute(1, 2, 0, 3).contiguous().view(batch, n_x, -1) # 3.Concat
        x_hid = x_hid.view(n_head, batch, n_x, d_k).permute(1, 2, 0, 3).contiguous().view(batch, n_x, -1) # 3.Concat
        attn = attn.view(n_head, batch, d_x_).permute(1, 2, 0).contiguous().view(batch, n_head, -1) # 3.Concat
     
        output = self.fc_o(output) # 4.
        x_hid = self.fc_o(x_hid)
    
        return attn, torch.squeeze(output,1), torch.squeeze(x_hid,1)

class scBALFModel(nn.Module):
    def __init__(self, d_x, d_k, hid_dim_x_0, hid_dim_x_1, hid_dim_cat, 
                 dropout_rate = 0.1, mask_k = 0, mha = False):
        super().__init__()        
        self.mha = mha
        self.n_emb = d_k
        self.input_dim = d_x
        self.hid_dim_cat = hid_dim_cat
        self.hid_dim_x_0 = hid_dim_x_0
        self.hid_dim_x_1 = hid_dim_x_1
        self.mask_k = mask_k
        #self.lin1 = nn.Linear(self.n_emb + self.n_emb, self.n_emb)
        #self.lin2 = nn.Linear(self.n_emb + self.input_dim, self.hid_dim)
        if mha:
            n_head = 5            
            d_k_per_head = 20
            self.crossattn = MultiHeadCrossAttention(n_head, d_x, d_k, d_k_per_head,  d_k)
        else:                
            self.crossattn = CrossAttention(d_x = self.hid_dim_x_1, d_k = self.n_emb)
        
        
        self.lin0_x = nn.Linear(self.input_dim, self.hid_dim_x_0)
        self.lin1_x = nn.Linear(self.hid_dim_x_0,self.hid_dim_x_1)
        self.lin1_k = nn.Linear(self.n_emb, self.hid_dim_cat)
        
        if self.mask_k == 1:
            self.lin2_x = nn.Linear(self.n_emb, self.hid_dim_cat) 
            self.bn2_x = nn.BatchNorm1d(self.n_emb)
            self.bn2 = nn.BatchNorm1d(self.hid_dim_cat)
            self.clf = nn.Linear(self.hid_dim_cat,3) 
        else: 
            self.lin2_x = nn.Linear(self.n_emb + self.hid_dim_x_1, self.hid_dim_cat)  
            self.bn2_x = nn.BatchNorm1d(self.n_emb + self.hid_dim_x_1)
            self.bn2 = nn.BatchNorm1d(2* self.hid_dim_cat)
            self.clf = nn.Linear(2* self.hid_dim_cat,3) 
            
        
        
        self.bn0_x = nn.BatchNorm1d(self.hid_dim_x_0)
        self.bn1_x = nn.BatchNorm1d(self.hid_dim_x_1)       
        
        self.bn1_k = nn.BatchNorm1d(self.n_emb)
        #self.bn3_x = nn.BatchNorm1d(self.hid_dim)
        self.drops = nn.Dropout(dropout_rate)
        

    def forward(self, x, k):     
        
        x = self.lin0_x(x) # 
        x = self.drops(x)  
        x = self.bn0_x(x)    

        x = F.leaky_relu(self.lin1_x(x)) # MLP
        x = self.drops(x)  
        x = self.bn1_x(x)        
        
        
        attn, k_hid, x_hid = self.crossattn(x, k)
        #x_hid = torch.cat([k_hid, x_hid],1) # concat embedding
        if self.mask_k== 1:
            x_hid = self.drops(x_hid)        
            x_hid = self.bn2_x(x_hid)
            x_hid = F.leaky_relu(self.lin2_x(x_hid)) # batch * n_emb    
        else: 
             # two flows K flow
            k_hid = self.drops(k_hid)
            k_hid = self.bn1_k(k_hid)
            k_hid = F.leaky_relu(self.lin1_k(k_hid))        
            # two flows X flow
            x_hid = torch.cat([x_hid, x],1)  # skip layer  
            x_hid = F.leaky_relu(self.lin2_x(x_hid))
        
        if self.mask_k == 1:
            x_hid = self.drops(x_hid)
            x_hid = self.bn2(x_hid)
            x_hid = self.clf(x_hid)  
        else:    
            x_hid = torch.cat([k_hid, x_hid],1)  # skip layer
            x_hid = self.drops(x_hid)
            x_hid = self.bn2(x_hid)
            x_hid = self.clf(x_hid)  
            
            #x_hid = F.relu(self.lin2(x_hid))        
            #x_hid = self.drops(x_hid)        
            #x_hid = self.bn3(x_hid)        
              
        return x_hid,attn
    

class scBALFModel_shallow(nn.Module):
    def __init__(self, d_x, d_k, dropout_rate = 0.1, mask_k = 0, mha = False):
        super().__init__()        
        self.mha = mha
        self.n_emb = d_k
        self.input_dim = d_x
        #self.hid_dim_cat = hid_dim_cat
        #self.hid_dim_x_0 = hid_dim_x_0
        #self.hid_dim_x_1 = hid_dim_x_1
        self.mask_k = mask_k
        #self.lin1 = nn.Linear(self.n_emb + self.n_emb, self.n_emb)
        #self.lin2 = nn.Linear(self.n_emb + self.input_dim, self.hid_dim)
        if mha:
            n_head = 5            
            d_k_per_head = 20
            self.crossattn = MultiHeadCrossAttention(n_head, d_x, d_k, d_k_per_head,  d_k)
        else:                   
            self.crossattn = CrossAttention(d_x = self.input_dim, d_k = self.n_emb)
        
        
        
        #self.lin1_x = nn.Linear(self.hid_dim_x_0,self.hid_dim_x_1)
        #self.lin1_k = nn.Linear(self.n_emb, self.hid_dim_cat)
        
        if self.mask_k == 1:
            #self.lin0 = nn.Linear(self.n_emb, self.hid_dim_cat)
            self.bn = nn.BatchNorm1d(self.n_emb)
            self.clf = nn.Linear(self.n_emb,3) 
        else: 
            #self.lin0 = nn.Linear(2*self.n_emb, self.hid_dim_cat)
            self.bn = nn.BatchNorm1d(2*self.n_emb )
            self.clf = nn.Linear(2* self.n_emb,3) 
            
                
        #self.bn1_k = nn.BatchNorm1d(self.n_emb)
        #self.bn3_x = nn.BatchNorm1d(self.hid_dim)
        self.drops = nn.Dropout(dropout_rate)
        

    def forward(self, x, k):         
        attn, k_hid, x_hid = self.crossattn(x, k)
        #x_hid = torch.cat([k_hid, x_hid],1) # concat embedding
        if self.mask_k== 1:
            x_hid = self.drops(x_hid)        
            x_hid = self.bn(x_hid)
            x_hid = self.clf(x_hid)    
        else: 
             # two flows K flow
            x_hid = torch.cat([k_hid, x_hid],1)
            x_hid = self.drops(x_hid)        
            x_hid = self.bn(x_hid)
            x_hid = self.clf(x_hid)  
        return x_hid,attn