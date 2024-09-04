import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchtext
import torchdata


import spacy
import tqdm
import evaluate
import datasets

from torchtext.vocab import build_vocab_from_iterator, GloVe, vocab, Vectors
from torch.utils.data import Sampler, Dataset
from torchtext.data.utils import get_tokenizer
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, Dataset
from torchtext.data import get_tokenizer


from ANNdebug import CNNparamaterStats,FCNparameterStats, hook_prnt_activations, hook_prnt_activation_norms
from ANNdebug import hook_prnt_inputs, hook_prnt_weights_grad_stats, callback_prnt_allweights_stats, callback_prnt_allgrads_stats
from ANNdebug import callback_prnt_weights_stats, hook_prnt_inputs_stats, hook_prnt_activations_stats, hook_prnt_inputs_norms, hook_return_activations, hook_return_inputs

import seaborn as sns
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')
import plotly.express as px
import plotly.graph_objects as go


import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from yellowbrick.regressor import PredictionError, ResidualsPlot

import copy
import random
import time
import sys
import os
import datetime
import logging
logging.raiseExceptions = False
import logging.handlers
from packaging import version

import collections
import unicodedata
import unidecode
import string
import re


def make_src_mask_transformerencoder(src, embed_dims=False):
    """
    Here assuming padding is always a 0
    src = [batch size, src len] or with embed dims
    src = [batch size, src len, embed_dims]
    """

    src_mask = (src == 0) 

    

    

    
    if embed_dims:
        src_mask = src_mask[:,:,0]
        
    return src_mask

def make_src_mask_scratch(src, embed_dims=False):
    """
    Here assuming padding is always a 0
    src = [batch size, src len] or with embed dims
    src = [batch size, src len, embed_dims]
    """
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    

    

    
    if embed_dims:
        src_mask = src_mask[:,:,:,:,0]
        
    return src_mask

class TransformerEncoderclassifier(nn.Module):

    def __init__(self , transformermodel, device, model_dims = 768, num_labels= 1):
        
        """
        num_labels gets the last out_features of linear layer, for binary choose 1, for multiclass choose 3 or more
        
        """
        super().__init__()
        self.device = device
        self.net = transformermodel
        self.num_labels = num_labels
        self.model_dims = model_dims
        self.linear1 = nn.Linear(self.model_dims,512)
        self.linear2 = nn.Linear(512,self.num_labels)
    
    def src_mask(self, src, embed_dims=True):
    
        """
        Here assuming padding is always a 0
        src = [batch size, src len] or with embed dims
        src = [batch size, src len, embed_dims]
        """

        src_mask = (src == 0).to(self.device) 

        

        


        if embed_dims:
            src_mask = src_mask[:,:,0]

        return src_mask
    
    def src_mean_mask(self, src, embed_dims=True):
        
        """
        Here assuming padding is always a 0
        """
        
    

     

        
        src_mask = (src != 0).to(self.device)

        

        

        if embed_dims:
            src_mask = src_mask[:,:,0]

        return src_mask
    
    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        
    def forward(self,src):
        
        src_mask = self.src_mask(src,embed_dims=True)
        mean_mask = self.src_mean_mask(src,embed_dims=True)
        
        x = self.net(src, src_key_padding_mask = src_mask) 

        x = self.mean_pooling(x,mean_mask) 

        
        x = self.linear1(x)
        x = nn.Dropout(p=0.1)(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x= x.squeeze() 

        return x


class TransformerFTseqclassifier(nn.Module):
    """
     Due to selfattn , cls ( the 0th element) contains info about other tokens in the sequence. So we just use cls token vector 
     as a represenation of the entire sequence and feed to classifier head.  
     
    """
    def __init__(self , transformermodel, device, num_labels= 1):
        
        """
        num_labels gets the last out_features of linear layer, for binary choose 1, for multiclass choose 3 or more
        
        """
        super().__init__()
        self.device = device
        self.net = transformermodel
        self.num_labels = num_labels
        self.linear1 = nn.Linear(768,512)
        self.linear2 = nn.Linear(512,self.num_labels)
        
        
    def forward(self,x,attn_mask):
        
        last_hidden_state = self.net(x,attention_mask = attn_mask).last_hidden_state
        x =  last_hidden_state[:,0]
        
        x = self.linear1(x)
        x = nn.Dropout(p=0.2)(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x= x.squeeze() 

        return x
        
        
class TransformerEncoder(nn.Module):
    """
    in hierarchical encoders, the dims of the first encoder are the input and hid dims of the second encoder
    so if first encoder was a distilbert, the input_dim=hid_dim of second encoder = 768
    """
    
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()

        self.device = device
        
        



    
        self.position_embedding = nn.Embedding(max_length, hid_dim) 

        

        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,n_heads,pf_dim,dropout,device) for _ in range(n_layers)])
        

        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        

        

        

        

        

        

        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) 

        

        
        

        

        

        
        




        src = self.dropout((src * self.scale) + self.position_embedding(pos)) 

        
        

        
       

    
        for layer in self.layers: 

            src = layer(src, src_mask)
            
        

            
        return src
    

class EncoderSelfAttn(nn.Module):
    
    def __init__(self,input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()

        self.device = device
        self.token_embedding = nn.Embedding(input_dim, hid_dim)
        self.position_embedding = nn.Embedding(max_length, hid_dim) 

        

        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,n_heads,pf_dim,dropout,device) for _ in range(n_layers)])
        

        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        

        

        

        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device) 

        

        
        

        

        

        
        

                
        src = self.dropout((self.token_embedding(src) * self.scale) + self.position_embedding(pos)) 

        
        

        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        

            
        return src
    
    
class EncoderLayer(nn.Module):
    
    def __init__(self, hid_dim, n_heads, pf_dim,  dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        

        
        

        

        

        

        

        

        

        

                
        

        _src, _ = self.self_attention(src, src, src, src_mask)  

        


        
        

        src = self.self_attn_layer_norm(src + self.dropout(_src)) 

        
        

        
        

        _src = self.positionwise_feedforward(src)
        
        

        src = self.ff_layer_norm(src + self.dropout(_src))
        
        

        
        return src

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, hid_dim, n_heads, dropout, device):
        
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None): 

        
        batch_size = query.shape[0]
        
        

        

        

                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        

        

        

        
        

        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        

        

        

                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale 

        
        

        
        if mask is not None: 

            

            

            energy = energy.masked_fill(mask == 0, -1e10) 

        
        attention = torch.softmax(energy, dim = -1)
                
        

                
        x = torch.matmul(self.dropout(attention), V) 

        
        

        
        

        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        

        
        x = x.view(batch_size, -1, self.hid_dim)
        
        

        
        x = self.fc_o(x) 

        
        

        
        return x, attention
    
class PositionwiseFeedforwardLayer(nn.Module):
    
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) 

        
    def forward(self, x):
        
        

        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        

        
        x = self.fc_2(x)
        
        

        

        x = self.dropout2(x)
        
        return x
    
class DecoderSelfAttn(nn.Module):
    
    def __init__(self, output_dim,hid_dim,n_layers,n_heads,pf_dim,dropout,device,max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.token_embedding = nn.Embedding(output_dim, hid_dim) 

        self.position_embedding = nn.Embedding(max_length, hid_dim) 

        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim,n_heads,pf_dim,dropout, device)  for _ in range(n_layers)]) 

        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device) 

        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        

        

        

        

                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        

            
        trg = self.dropout((self.token_embedding(trg) * self.scale) + self.position_embedding(pos))   
         

        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        

        

        
        output = self.fc_out(trg)
        
        

            
        return output, attention       
        
class DecoderLayer(nn.Module):
    
    def __init__(self, hid_dim,n_heads,pf_dim,dropout,device):
     
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        

        

        

        

        
        

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        

        

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        

        
        

        
        

        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        

        

        
        return trg, attention

class Seq2SeqSelfAttn(nn.Module):
    
    def __init__(self,encoder,decoder,src_pad_idx,trg_pad_idx,device ):                
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        

        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)

        


        return src_mask
    
    def make_trg_mask(self, trg):
        
        

        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        
        

        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        

            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        

        
        return trg_mask

    def forward(self, src, trg):
        
        

        

                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        

        

        
        enc_src = self.encoder(src, src_mask)
        
        

                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        

        

        
        return output, attention