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



def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def log_setup(loglvl,logtodisk,write_file):
    logger = logging.getLogger('myLogger')
    formatter=logging.Formatter('%(asctime)s %(name)s %(process)d %(message)s')
    if logtodisk:
        filehandler=logging.FileHandler(write_file)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    if loglvl=='DEBUG':
        logger.setLevel(logging.DEBUG)
        return logger
    else:
        logger.setLevel(logging.INFO)
        return logger
    

    
class custReshape(nn.Module):
    def __init__(self, *args):
        
        """
               self.decoder = nn.Sequential(
                torch.nn.Linear(2, 3136),
                custReshape(-1, 64, 7, 7) 

        so args here are = (-1, 64, 7, 7)
        """
        super().__init__()
        self.shape = args

    def forward(self, x): 

        

        return x.view(self.shape)
    
class permuteTensor(nn.Module):
    def __init__(self, *args):
        super().__init__()
        """
        call like permuteTensor(0,2,1) to rearrange the the dimensions
        """
        self.dims = args
     
    def forward(self, x):
        return x.permute(self.dims)
    
class globalMaxpool(nn.Module):
    def __init__(self, dim =2):
        super().__init__()
        """
        returns one token which has the max vector in a sequence of tokens 
        dim  = the dimension which corresponds to sequence length
        
        """
        self.dim = dim
     
    def forward(self, x):
        x,_= torch.max(x, self.dim)
        return x
    
class concatTwotensors(nn.Module):
    
    def __init__(self, dim =None):
        super().__init__()
        """
        concats tensors over specified dims
        """
        self.dim = dim
     
    def forward(self,tens1,tens2):
        return torch.cat((tens1,tens2), dim = self.dim)
    
class concatThreetensors(nn.Module):
    def __init__(self, dim =None):
        super().__init__()
        """
        concats tensors over specified dims
        """
        self.dim = dim
     
    def forward(self,tens1,tens2,tens3):
        return torch.cat((tens1,tens2,tens3), dim = self.dim)

    
class squeezeOnes(nn.Module):
    def __init__(self, *args):
        super().__init__()    
    
    def forward(self, x):
        return x.squeeze()
    
class standin(nn.Module):
    def __init__(self):
        super().__init__()    
    
    def forward(self, x):
        return x   
    
class unsqueezeOnes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.unsqueeze(self.dim)
    
    
class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.w = args[0]
        self.h = args[1]

    def forward(self, x):
        return x[:, :, :self.w, :self.h] 



class Linearhc(nn.Module): 
    
    """
    This is a linear layer which takes output/hidden/cell of lstm with (NOTE) seq len 1 (decoder in seq2seq)
    and returns output passed throgh linear layer and hidden/cell. NOTE the squeeze is over seq len as it is one
    """
   
    
    def __init__(self,infeatures,outfeatures):
        super().__init__()
        self.linearhc = nn.Linear(infeatures,outfeatures)
        
    def forward(self,x):
        output, (hidden, cell) = x
        

        

        output = self.linearhc(output.squeeze(1)) 

        
        return output, (hidden, cell)

class Linearhchiddencell(nn.Module): 
    
    """
    This is a linear layer which takes output/hidden/cell of lstm with
    and returns hidden and cell passed throgh linear layer and output untouched. 
    """
   
    
    def __init__(self,infeatures,outfeatures):
        super().__init__()
        self.linearhc = nn.Linear(infeatures,outfeatures)
        
    def forward(self,x):
        output, (hidden, cell) = x
        

        hidden = self.linearhc(hidden)
        cell = self.linearhc(cell)

        return output, (hidden, cell)
    
class GRULinearhchidden(nn.Module): 

    
    def __init__(self,infeatures,outfeatures):
        super().__init__()
        self.linearhc = nn.Linear(infeatures,outfeatures)
        
    def forward(self,x):
        output, hidden = x
        

        hidden = self.linearhc(hidden)

        return output, hidden
        
class UnpackpackedOutput(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        packed_outputs, hidden = x
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs,batch_first=True)
        return outputs
        
class UnpackpackedOutputHidden(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        packed_outputs, hidden = x
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden    

class activationhc(nn.Module): 
    
    """
    This takes in the output of lstm and only processes the hc through act func and returns out, (h,c)
    """
    def __init__(self,actfunc):
        super().__init__()
        self.actfunc = actfunc
        
    def forward(self,x):
        output, (hidden, cell) = x
        hidden = self.actfunc(hidden)
        cell = self.actfunc(cell)
        
        return output, (hidden, cell)
    
class activationh(nn.Module): 

    def __init__(self,actfunc):
        super().__init__()
        self.actfunc = actfunc
        
    def forward(self,x):
        output,hidden = x
        hidden = self.actfunc(hidden)

        return output, hidden
        
class Bidirectionfullprocess(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        
        output, (hidden, cell) = x
        

        

        try:
            hidden = hidden.view(int(hidden.shape[0]/2),2,hidden.shape[1], hidden.shape[2])
            

            hidden = hidden[-1] 

            

            cell = cell.view(int(cell.shape[0]/2),2,cell.shape[1], cell.shape[2])
            cell = cell[-1] 

        except Exception as e:
            print ("debug: ", output.shape, hidden.shape, cell.shape)
            sys.exit(0)
            
        h_fwd, h_bwd = hidden[0], hidden[1]
        

        h_n = torch.cat((h_fwd, h_bwd), dim=1)
        

        c_fwd, c_bwd = cell[0], cell[1]
        c_n = torch.cat((c_fwd, c_bwd), dim=1)  
        
        return output, (h_n, c_n) 
        

        


class GRUBidirectionfullprocess(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        
        output, hidden = x
        

        

        try:
            hidden = hidden.view(int(hidden.shape[0]/2),2,hidden.shape[1], hidden.shape[2])
            

            hidden = hidden[-1] 

            

        except Exception as e:
            print ("debug: ", output.shape, hidden.shape, cell.shape)
            sys.exit(0)
            
        h_fwd, h_bwd = hidden[0], hidden[1]
        

        h_n = torch.cat((h_fwd, h_bwd), dim=1)
        


        
        return output, h_n
        

        

        
class BidirectionextractHiddenfinal(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        try:
            hidden = hidden.view(int(hidden.shape[0]/2),2,hidden.shape[1], hidden.shape[2])
            hidden = hidden[-1] 

        except Exception as e:
            print ("debug: ", output.shape, hidden.shape)
            sys.exit(0)
        return hidden
    
class hiddenBidirectional(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        h_fwd, h_bwd = x[0], x[1]
        h_n = torch.cat((h_fwd, h_bwd), dim=1)
        return h_n

    
class BidirectionextractHCfinal(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        try:
            hidden = hidden.view(int(hidden.shape[0]/2),2,hidden.shape[1], hidden.shape[2])
            hidden = hidden[-1] 

            cell = cell.view(int(cell.shape[0]/2),2,cell.shape[1], cell.shape[2])
            cell = cell[-1] 

        except Exception as e:
            print ("debug: ", output.shape, hidden.shape, cell.shape)
            sys.exit(0)
        return (hidden, cell)    

class hcBidirectional(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, h,c):
        h_fwd, h_bwd = h[0], h[1]
        h_n = torch.cat((h_fwd, h_bwd), dim=1)
        c_fwd, c_bwd = c[0], c[1]
        c_n = torch.cat((c_fwd, c_bwd), dim=1)        
        return (h_n, c_n)
    
class hcHiddenonlyBidirectional(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, h,c):
        h_fwd, h_bwd = h[0], h[1]
        h_n = torch.cat((h_fwd, h_bwd), dim=1)      
        return h_n

class LSTMhc(nn.Module):
    
    def __init__(self,input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False):
        super().__init__()
        self.lstmhc = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        
    def forward(self,x,hc):

        output, (hidden, cell) = self.lstmhc(x,hc)
        return output, (hidden, cell)
    

class UnidirectionalextractOutput(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        return output.squeeze(0)
           

        
class UnidirectionextractHiddenCell(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        return (hidden,cell)

class UnidirectionextractHidden(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        return hidden
 
    
class UnidirectionextractHiddenfinal(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        output, (hidden, cell) = x
        try:
            hidden = hidden.view(hidden.shape[0],1,hidden.shape[1], hidden.shape[2])
            hidden = hidden[-1] 

        except Exception as e:
            print ("debug: ", output.shape, hidden.shape)
            sys.exit(0)
        return hidden
    


        
class hiddenUnidirectional(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.squeeze(0)


class configuration(object): 
    
    def __init__(self):
        pass
    
    def configureRNN(self, rnnlayers = {}, params = {}):
        
        """
        rnnlayers = {1:embed,2:lstm, 3:fcn}
        params = {"batch_size":128,"pretrained_embeddings" : False}
        """
        self.rnnnet = rnnlayers, params

    def configureFCN(self,layers={1:[4,12],2:[12,12],3:[12,3]}):
        
        """
           FCNlayer = {1:linear1, 2:relu, 3:linear2, 4:relu, 5:linear3, 6:relu, 7:linear4,
                8:sigmoid} 

        """
        self.fcnnet = layers
        
    def configureCNN1d(self,convlayers = {}, inputsize = 0):
        
        assert inputsize > 0, "Check input size" 
        
        outsize = {} 

        finaloutsize =[]
        insize = {}
        

        insize[1] = inputsize 

        outsize[1] = inputsize 

        finaloutchannels = 0
        conv_key = list(convlayers.keys())

        for i in convlayers:




            if 'Conv1d' in str(type(convlayers[i])):




                outsize[i] = np.floor((inputsize + 2*convlayers[i].padding[0] - convlayers[i].kernel_size[0])/convlayers[i].stride[0]) +1
                

                finaloutchannels = convlayers[i].out_channels  
                finaloutsize = outsize[i]
                inputsize = outsize[i]
                insize[i+1] = outsize[i]


        
            if 'pooling' in str(type(convlayers[i])):

                

                outsize[i] = np.floor((inputsize -convlayers[i].kernel_size)/convlayers[i].stride) +1
                finaloutsize = outsize[i]
                inputsize = outsize[i]
                insize[i+1] = outsize[i] 
                


        self.convoutsize = finaloutsize
        self.finaloutconvchannels = finaloutchannels
        self.convsizeperlayer = outsize
        print (" Final outsize: ", self.convoutsize)
        
        self.cnn1dnet = convlayers 

    def configureCNN(self, convlayers = {},deconvlayers = {} ,inputsize = [], deconv_inputsize =[]):
        """
        NOTE: one conv or/and deconv layer per configurecnn, where each have to be consecutive in order- not sure about this anymore
        inputsize [width, height]
        conv blocks convolution -> pool -> batchnorm -> relu
        convlayer = {1:conv1,2:relu, 3:pool1, 
             4:conv2,5:relu, 6:pool2}
        deconvlayer = {7:deconv1,8:relu,
               9:deconv2}
        
        Returns vars : self.cnnnet, self.fcnnet, self.fcin, self.convoutimgsize(list for each conv block), self.finaloutconvchannels,self.deconvfcin, self.deconvoutimgsize(list for each conv block), self.finaloutdeconvchannels
        
        """

        

        if convlayers:
            outimgsize = {} 

            finaloutimgsize =[]
            inimgsize = {}
            

            inimgsize[1] = inputsize 

            outimgsize[1] = inputsize 

            finaloutchannels = 0
            conv_key = list(convlayers.keys())


            for i in convlayers:

    


                if 'conv' in str(type(convlayers[i])):
    

                    outimgsize[i] = [np.floor((inputsize[0] + 2*convlayers[i].padding[0] -
                                               convlayers[i].kernel_size[0])/convlayers[i].stride[0]) +1,
                                      np.floor((inputsize[1] + 2*convlayers[i].padding[0] -
                                               convlayers[i].kernel_size[0])/convlayers[i].stride[0]) +1
                                               ]
                    

                    finaloutchannels = convlayers[i].out_channels  
                    finaloutimgsize = outimgsize[i]
                    inputsize = outimgsize[i]
                    inimgsize[i+1] = outimgsize[i]
    


                if 'pool' in str(type(convlayers[i])):

                    

                    outimgsize[i] = [np.floor((inputsize[0] -convlayers[i].kernel_size)/convlayers[i].stride) +1,
                                          np.floor((inputsize[1] -convlayers[i].kernel_size)/convlayers[i].stride) +1
                                           ]
                    finaloutimgsize = outimgsize[i]
                    inputsize = outimgsize[i]
                    inimgsize[i+1] = outimgsize[i]

            self.fcin = int(finaloutchannels*finaloutimgsize[0]*finaloutimgsize[1])
            self.convoutimgsize = finaloutimgsize
            self.finaloutconvchannels = finaloutchannels
            self.convimgsizeperlayer = outimgsize

        
        
        

        if deconvlayers:
            deconv_key = list(deconvlayers.keys())
            finaloutimgsize = []
            outimgsize = {} 

            inimgsize = {}
            finaloutchannels = 0

            if not deconv_inputsize: 

                deconv_inputsize = self.convoutimgsize

        

            inimgsize[min(deconv_key)] = deconv_inputsize 

            outimgsize[min(deconv_key)] = deconv_inputsize 


            for i in deconvlayers:




                if 'conv' in str(type(deconvlayers[i])): 

                    outimgsize[i] = [((deconv_inputsize[0] -1)*deconvlayers[i].stride[0] - 2*deconvlayers[i].padding[0] + (deconvlayers[i].kernel_size[0] -1) + deconvlayers[i].output_padding[0]+1),
                                     ((deconv_inputsize[1] -1)*deconvlayers[i].stride[1] - 2*deconvlayers[i].padding[1] + (deconvlayers[i].kernel_size[1] -1) + deconvlayers[i].output_padding[1]+1)]


                    

                    finaloutchannels = deconvlayers[i].out_channels
                    finaloutimgsize = outimgsize[i]
                    deconv_inputsize = outimgsize[i]
                    inimgsize[i+1] = outimgsize[i]

                if 'pool' in str(type(deconvlayers[i])):


                    outimgsize[i] = [np.floor((deconv_inputsize[0] -deconvlayers[i].kernel_size)/deconvlayers[i].stride) +1,
                                          np.floor((deconv_inputsize[1] -deconvlayers[i].kernel_size)/deconvlayers[i].stride) +1
                                           ]
                    finaloutimgsize = outimgsize[i]
                    deconv_inputsize = outimgsize[i]
                    inimgsize[i+1] = outimgsize[i]
                


            self.deconvfcin = int(finaloutchannels*finaloutimgsize[0]*finaloutimgsize[1])
            self.deconvoutimgsize = finaloutimgsize
            self.finaloutdeconvchannels = finaloutchannels
            self.deconvimgsizeperlayer = outimgsize
            
            
        self.cnnnet = convlayers, deconvlayers
        
    def conv(self,inchannels = 0, outchannels = 0, kernel_size = 5, stride = 1, padding = 0):
        
        return nn.Conv2d(inchannels, outchannels, kernel_size, stride, padding)
    
    def maxpool(self,kernel_size =2, stride = None):
        
        return nn.MaxPool2d(kernel_size, stride = stride)
    
    def avgpool(self,kernel_size =2, stride = None):
        
        return nn.AvgPool2d(kernel_size, stride = stride)
    
    def conv1d(self,inchannels = 0, outchannels = 0, kernel_size = 5, stride = 1, padding = 0):
        return nn.Conv1d(inchannels, outchannels, kernel_size, stride, padding)
    
    def maxpool1d(self,kernel_size =2, stride = None):
        return nn.MaxPool1d(kernel_size, stride = stride)
        
    
    def batchnorm2d(self,channels =0):
        
        return nn.BatchNorm2d(channels)
    
    def convtranspose(self,inchannels = 0, outchannels = 0, kernel_size = 5, stride = 1, padding = 0,output_padding =0):
        return nn.ConvTranspose2d(inchannels, outchannels, kernel_size, stride, padding,output_padding)
    
    def batchnorm1d(self,num_features=0):
        return nn.BatchNorm1d(num_features =  num_features) 

    
    def relu(self, inplace = False):
        return torch.nn.ReLU(inplace = inplace)
    
    def tanh(self):
        return torch.nn.Tanh()
    
    
    def leaky_relu(self,negative_slope=0.01, inplace = False):
        return nn.LeakyReLU(negative_slope=negative_slope, inplace = inplace)   

    
    def sigmoid(self):
        return nn.Sigmoid() 

    
    def dropout(self,p=0.5):
        return nn.Dropout(p=p) 

    
    def dropout1d(self,p=0.25):
        return nn.Dropout1d(p=p)
    
    def dropout2d(self,p=0.25):
        return nn.Dropout2d(p=p)
    
    def linear(self, infeatures = 0, outfeatures = 0):
        return nn.Linear(infeatures,outfeatures)
    
    def flatten(self, start_dim=1, end_dim=-1):
        return nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    
    def unflatten(self, dim, unflattened_size):
        return nn.Unflatten(dim, unflattened_size) 

    
    def embeddings(self, num_embeddings, embedding_dim):
        return nn.Embedding(num_embeddings, embedding_dim)
    
    def pretrained_embeddings(self, weights, freeze =  True):
        return nn.Embedding.from_pretrained(weights, freeze=freeze)
    
    def lstm(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False):
        return nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def gru(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0, bidirectional=False):
        return nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def rnn(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=False, dropout=0.0, bidirectional=False):
        return nn.RNN(input_size, hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    
    def modulelist(self,modls):
        """
        pass modules in a python list
        """
        tmpmodls = nn.ModuleList()
        for m in modls:
            tmpmodls.append(m)
        return tmpmodls 













class VAE(nn.Module):
    def __init__(self, encoder = None, decoder = None, fcin = 0, zdims =2):
        """
        provide full decoder and encoder net as ANN objects
        zmean and zlogvar can nets themselves or use single layer linear with fcin and zdims values
        """
        super().__init__()
        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()
        for i in encoder:
            self.encoder[str(i)] = encoder[i]
        
        self.z_mean = nn.Linear(fcin, zdims)
        self.z_log_var = nn.Linear(fcin, zdims)
        
        for i in decoder:
            self.decoder[str(i)] = decoder[i]
     
    def embedding(self,x):
        for i in self.encoder:
            x = self.encoder[i](x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x) 

        encoding = self.reparameterization(z_mean, torch.exp(0.5 * z_log_var)) 

        return encoding
    
    def generate_from_latent_space(self,x):
        for i in self.decoder:
            x = self.decoder[i](x)
        return x
 
        
    def reparameterization(self, mean, exponentvar): 

        epsilon = torch.randn_like(exponentvar) 

        z = mean + exponentvar*epsilon                          
        return z 

    
    def forward(self, x):
        for i in self.encoder:




            x = self.encoder[i](x)
           
            
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x) 



        z = self.reparameterization(z_mean, torch.exp(0.5 * z_log_var)) 



        
        for c,i in enumerate(self.decoder):
            if c ==0:




                xhat= self.decoder[i](z)
            else:




                xhat = self.decoder[i](xhat)


        
        return z, z_mean, z_log_var, xhat    
    

class ANN(nn.Module):   
    
    def __init__(self, confobj ={}):
        """
        confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() 

        self.layers = nn.ModuleDict()
        self.layersdict = {}
        if confobj:
            for network in confobj:
                if network == 'CNN':
                    self.CNN =  True
                    for cfg in confobj[network]:
                        self.cnn(cfg.cnnnet[0],cfg.cnnnet[1])
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)

            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})






    def cnn(self, convlayers={}, deconvlayers ={}):

        assert len(convlayers) >0 , "Check Layers"

        for i in convlayers:
            self.layersdict[i] = convlayers[i] 

        for i in deconvlayers:                   
            self.layersdict[i] = deconvlayers[i] 

                 
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" 





        for i in fclayers:
            self.layersdict[i] = fclayers[i]

    def forward(self,x):

        for i in self.layers: 






            x = self.layers[i](x)
        return x


class CNN1D(nn.Module):   
    
    def __init__(self, confobj ={}):
        """
        confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() 

        self.layers = nn.ModuleDict()
        self.layersdict = {}
        if confobj:
            for network in confobj:
                if network == 'CNN':
                    self.CNN =  True
                    for cfg in confobj[network]:
                        self.cnn(cfg.cnn1dnet)
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)

            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})




    def cnn(self, convlayers={}):

        assert len(convlayers) >0 , "Check Layers"

        for i in convlayers:
            self.layersdict[i] = convlayers[i] 
                 
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" 



        for i in fclayers:
            self.layersdict[i] = fclayers[i]

    def forward(self,x):

        for i in self.layers: 






            x = self.layers[i](x)
        return x
    
    
class RNN_classification(nn.Module):   
    
    def __init__(self,confobj ={}):
        
        """confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() 

        self.layers = nn.ModuleDict()
        self.layersdict = {}


        self.bidirectional = None
        self.directions = None
        self.batch_size = None


        self.hidden_dims = None
        self.num_layers = None
        self.rnnlayeridx = None
        if confobj:
            for network in confobj:
                if network == 'RNN' or network == 'LSTM':
                    self.RNN =  True
                    for cfg in confobj[network]:




                        self.rnn(cfg.rnnnet[0])               
              
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)
                       
            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})

            if self.RNN:
                params = cfg.rnnnet[1]
            for p in params:
                if "batch_size" in p:
                    self.batch_size = params[p]
                    
            

            for l in self.layers:
                if 'rnn.RNN' in str(type(self.layers[l])) or 'rnn.LSTM' in str(type(self.layers[l])):
                    self.rnnlayeridx = l
                    self.hidden_dims = self.layers[l].hidden_size
                    self.num_layers = self.layers[l].num_layers
                    if self.layers[l].bidirectional:
                        self.bidirectional = True
                        self.directions =  2
                    else:
                        self.bidirectional = False
                        self.directions =  1
                        
            
    def rnn(self, rnnlayers = {}):
        assert len(rnnlayers) >0 , "Check Layers" 
        
        
        for i in rnnlayers:
            self.layersdict[i] = rnnlayers[i]  
                        
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" 
        



        for i in fclayers:
            self.layersdict[i] = fclayers[i]        

    def forward(self,x):

        for i in self.layers:
            
            

            

            x = self.layers[i](x)
            
            """
            if i == self.rnnlayeridx:
                
                output, (hidden, cell) = self.layers[i](x)
                hidden = hidden.view(self.num_layers,self.directions,self.batch_size,self.hidden_dims)
                hidden = hidden[-1] 

                if self.directions ==2:
                    h_fwd, h_bwd = hidden[0], hidden[1]
                    h_n = torch.cat((h_fwd, h_bwd), dim=1)
                    x = h_n.view(self.batch_size,self.hidden_dims*2)
                elif self.directions == 1:
                    x = hidden.view(self.batch_size,self.hidden_dims)
            
            else:
                
                x = self.layers[i](x)
            """
                
        return x 
    
    
class RNNhc(nn.Module):   
    
    

    

    def __init__(self,confobj ={}):
        
        """confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() 

        self.layers = nn.ModuleDict()
        self.layersdict = {}


        self.bidirectional = None
        self.directions = None
        self.batch_size = None


        self.hidden_dims = None
        self.num_layers = None
        self.rnnlayeridx = None
        if confobj:
            for network in confobj:
                if network == 'RNN' or network == 'LSTM':
                    self.RNN =  True
                    for cfg in confobj[network]:




                        self.rnn(cfg.rnnnet[0])               
              
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)
                       
            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})

            if self.RNN:
                params = cfg.rnnnet[1]
            for p in params:
                if "batch_size" in p:
                    self.batch_size = params[p]
                    
            

            for l in self.layers:
                if 'rnn.RNN' in str(type(self.layers[l])) or 'rnn.LSTM' in str(type(self.layers[l])):
                    self.rnnlayeridx = l
                    self.hidden_dims = self.layers[l].hidden_size
                    self.num_layers = self.layers[l].num_layers
                    if self.layers[l].bidirectional:
                        self.bidirectional = True
                        self.directions =  2
                    else:
                        self.bidirectional = False
                        self.directions =  1
                        
            
    def rnn(self, rnnlayers = {}):
        assert len(rnnlayers) >0 , "Check Layers" 
        
        
        for i in rnnlayers:
            self.layersdict[i] = rnnlayers[i]  
                        
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" 
        



        for i in fclayers:
            self.layersdict[i] = fclayers[i]        

    def forward(self,x, h= None, c = None, contextvec = None):
        
        

        
        

        

        

        

        

        

        for i in self.layers:

            if 'LSTM' in str(type(self.layers[i])) or 'RNN' in str(type(self.layers[i])):
                x= self.layers[i](x,(h,c))   
            else:
                x = self.layers[i](x)

        return x 

    
class RNNpacked(nn.Module):   

    def __init__(self,confobj ={}):
        
        """confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        everything including activation, dropout , batchnorms, flatten, reshape need to be in there.
        """
        super().__init__() 

        print ("RNNpacked class USED")
        self.layers = nn.ModuleDict()
        self.layersdict = {}


        self.bidirectional = None
        self.directions = None
        self.batch_size = None


        self.hidden_dims = None
        self.num_layers = None
        self.rnnlayeridx = None
        if confobj:
            for network in confobj:
                if network == 'RNN' or network == 'LSTM':
                    self.RNN =  True
                    for cfg in confobj[network]:




                        self.rnn(cfg.rnnnet[0])               
              
                if network == 'FCN':
                    self.FCN =  True
                    for cfg in confobj[network]:
                        self.fcn(cfg.fcnnet)
                       
            self.layers = nn.ModuleDict({str(k): self.layersdict[k] for k in sorted(self.layersdict)})

            if self.RNN:
                params = cfg.rnnnet[1]
            for p in params:
                if "batch_size" in p:
                    self.batch_size = params[p]
                if "pack" in p:
                    self.packidx = params[p]


                    
            

            for l in self.layers:
                if 'rnn.RNN' in str(type(self.layers[l])) or 'rnn.LSTM' in str(type(self.layers[l])):
                    self.rnnlayeridx = l
                    self.hidden_dims = self.layers[l].hidden_size
                    self.num_layers = self.layers[l].num_layers
                    if self.layers[l].bidirectional:
                        self.bidirectional = True
                        self.directions =  2
                    else:
                        self.bidirectional = False
                        self.directions =  1
                        
            
    def rnn(self, rnnlayers = {}):
        assert len(rnnlayers) >0 , "Check Layers" 
        
        for i in rnnlayers:
            self.layersdict[i] = rnnlayers[i]  
                        
            
    def fcn(self, fclayers={}):
        
        assert len(fclayers) >0 , "Check Layers" 
        



        for i in fclayers:
            self.layersdict[i] = fclayers[i]        

    def forward(self,x,x_len=None):
        
        

        
        

        

        

        

        

        

        assert isinstance(x_len, torch.Tensor), "This is a RNNpacked class, src len cannot be None"
        
        for i in self.layers:
            if i == self.packidx:
                x = nn.utils.rnn.pack_padded_sequence(x, x_len.cpu(),batch_first=True,enforce_sorted=False)
            else:
                x = self.layers[i](x)

        return x 
    
class decoderGRU_cho(nn.Module): 
    """
    Decoder with GRU implemented from https://arxiv.org/abs/1406.1078
    """
    
    def __init__(self,output_dim, emb_dim, hid_dim,num_layers, dropout = 0.0):
        
        """
        output_dims = output vocab size. This is used as the outfeatures in final linear layer
        
        """
        super().__init__() 

        
        self.hid_dim = hid_dim
        self.output_dim = output_dim 

        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers=num_layers, batch_first= True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)  

    def forward(self,input, hidden, contextvec = None):
        
        
        

        

        

        

        

        

        input = input.unsqueeze(1) 

        embedded = self.dropout(self.embedding(input)) 

        
        

        
        

        """
        [[batchsize , seqlen, cat (embed dims, context vec)]] as a number samples, each sample having some seq len ( number of words)
and each element in the sequence ( each token) having a number of features given by embed dims + context vec.
Intutiively this makes sence as now each input seq to the decoder whose next word is to be predicted by model carries info
about its own embeddings and also each word carries context vector info from encoder.
        """
        

        

        contextvec = contextvec.permute(1,0,2) 

        emb_con = torch.cat((embedded, contextvec), dim = 2) 

        output, hidden = self.rnn(emb_con, hidden)
        

       
    
    

        """
embed input , output of lstm = hidden since there is 1 token predic, encoder contextvec
embed input =  [batch, 1, embed dims], hidden = [1, batch size, hid dim], 
contextvec = [D*num_layers = 1, batch size, hidden dim] converted to [batch size,1, hidden dim]

This needs to be fed as input to linear layer so needs to be squeezed out of 1 and into general form of [batchs, features]
embedded.squeeze(1) = [batch, 1, embed dims],  hidden.squeeze(0) = [batch size, hid dim]
contextvec.squeeze(1) = [batch size, hid dim] 
output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), context.squeeze(1)), dim = 1)
concat over dim1  ouput =  [batchsize, emb dim + hid dim * 2]
        """
        output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), contextvec.squeeze(1)), dim = 1) 

        prediction = self.fc_out(output) 



        return prediction, hidden   

class decoder_cho(nn.Module): 
    """
    Decoder with LSTM implemented from https://arxiv.org/abs/1406.1078
    """
    
    def __init__(self,output_dim, emb_dim, hid_dim,num_layers, dropout = 0.0):
        
        """
        output_dims = output vocab size. This is used as the outfeatures in final linear layer
        
        """
        super().__init__() 

        
        self.hid_dim = hid_dim
        self.output_dim = output_dim 

        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, num_layers=num_layers, batch_first= True)
        

        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)  

    def forward(self,input, hc, contextvec = None):
        
        
        

        

        

        

        

        

        input = input.unsqueeze(1) 

        embedded = self.dropout(self.embedding(input)) 

        
        

        
        

        """
        [[batchsize , seqlen, cat (embed dims, context vec)]] as a number samples, each sample having some seq len ( number of words)
and each element in the sequence ( each token) having a number of features given by embed dims + context vec.
Intutiively this makes sence as now each input seq to the decoder whose next word is to be predicted by model carries info
about its own embeddings and also each word carries context vector info from encoder.
        """
        

        

        contextvec = contextvec.permute(1,0,2) 

        emb_con = torch.cat((embedded, contextvec), dim = 2) 

        output, (hidden, cell) = self.rnn(emb_con, (hc[0],hc[1]))
        

       
    
    

        """
embed input , output of lstm = hidden since there is 1 token predic, encoder contextvec
embed input =  [batch, 1, embed dims], hidden = [1, batch size, hid dim], 
contextvec = [D*num_layers = 1, batch size, hidden dim] converted to [batch size,1, hidden dim]

This needs to be fed as input to linear layer so needs to be squeezed out of 1 and into general form of [batchs, features]
embedded.squeeze(1) = [batch, 1, embed dims],  hidden.squeeze(0) = [batch size, hid dim]
contextvec.squeeze(1) = [batch size, hid dim] 
output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), context.squeeze(1)), dim = 1)
concat over dim1  ouput =  [batchsize, emb dim + hid dim * 2]
        """
        output = torch.cat((embedded.squeeze(1), hidden.squeeze(0), contextvec.squeeze(1)), dim = 1) 

        prediction = self.fc_out(output) 



        return prediction, (hidden, cell)         

class Attention(nn.Module):

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask = None):
        
        

        



        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.squeeze(0) 

        
        

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        


        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        


        attention = self.v(energy).squeeze(2)
  
        

        if mask is not None: 

            attention = attention.masked_fill(mask == 0, -1e10) 

        
        return F.softmax(attention, dim=1)

class decoderGRU_attn_bahdanau(nn.Module):
    
    """
    Decoder GRU for attention 
    """
    
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim 

        self.attention = attention 

        
        self.embedding = nn.Embedding(output_dim, emb_dim)

            
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim,num_layers=num_layers,batch_first=True)

        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        

        
    def forward(self, input, hc, encoder_outputs, mask =None):
             
        

        

        
        

        



        input = input.unsqueeze(1) 

        
        embedded = self.dropout(self.embedding(input)) 


        hidden = hc

        a = self.attention(hidden, encoder_outputs, mask) 

        


        a = a.unsqueeze(1) 


        weighted = torch.bmm(a, encoder_outputs) 

        

        


        
        rnn_input = torch.cat((embedded, weighted), dim = 2) 

        
        

        

        

        
        
        hidden = hidden.squeeze(0)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        

        

                                         
        

        

        
        

        

        


        

        assert (output.permute(1,0,2) == hidden).all()
        
        embedded = embedded.squeeze(1)  

        output = output.squeeze(1) 

        weighted = weighted.squeeze(1) 


        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        

        
        return prediction, hidden , a.squeeze(1) 

         

        


class decoder_attn_bahdanau(nn.Module):
    
    """
    Decoder LSTM for attention 
    """
    
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        
        super().__init__()

        self.output_dim = output_dim 

        self.attention = attention 

        
        self.embedding = nn.Embedding(output_dim, emb_dim)



        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim,num_layers=num_layers,batch_first=True)


        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        

        
    def forward(self, input, hc, encoder_outputs, mask = None):
             
        

        

        
        

        



        input = input.unsqueeze(1) 

        
        embedded = self.dropout(self.embedding(input)) 


        hidden = hc[0]
        cell = hc[1]


        a = self.attention(hidden, encoder_outputs, mask) 

        


        a = a.unsqueeze(1) 


        weighted = torch.bmm(a, encoder_outputs) 

        

        


        
        rnn_input = torch.cat((embedded, weighted), dim = 2) 

        
        

        

        

        
       
        
        hidden = hidden.squeeze(0) 
        cell = cell.squeeze(0)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0))) 

        

                                         
        

        

        
        

        

        

        

        

        assert (output.permute(1,0,2) == hidden).all()
        
        embedded = embedded.squeeze(1)  

        output = output.squeeze(1) 

        weighted = weighted.squeeze(1) 


        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        


        
        return prediction, (hidden, cell), a.squeeze(1) 

         

        
     
class Seq2SeqLSTMPacked(nn.Module):
    
    """
    encoder decoder are rnn/lstm networks
    usually we want hidden dims and layer on then to be the same
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    """    
    def __init__(self,encoder, decoder, params = {}):
        super().__init__() 

        self.encoder = encoder 
        self.decoder = decoder
        self.src_vocab_len = 0
        self.trg_vocab_len = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for p in params:




            if 'src_vocab_len' in p:
                self.src_vocab_len = params[p]
            if 'trg_vocab_len' in p:
                self.trg_vocab_len = params[p]    
            if 'device' in p:
                self.device = params[p]

        


        assert self.src_vocab_len> 0, "No src vocab len"
        assert self.trg_vocab_len> 0, "No trg vocab len"

    
    def forward(self, src, src_len =None, trg = None, teacher_forcing= 0.5): 

        """
        src = [ batch size, src seq len]
        src_len = [batch size]
        trg = [ batch size, trg seq len,]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        """
        assert isinstance(src_len, torch.Tensor), "This is a seq2seq packed class, src len cannot be None"
        
        if trg == None: 

            trg = torch.zeros((src.shape[0], 25)).fill_(0).long().to(self.device)
            

            

            assert teacher_forcing == 0, "Must be zero during inference"
            
        

        trg_len = trg.shape[1] 

        bs = trg.shape[0] 

        


                 
        outputs = torch.zeros(trg_len, bs, self.trg_vocab_len).to(self.device)
        
        hidden = self.encoder(src, src_len)

 


        context = hidden 

        


        inp = trg[:,0] 


        for t in range(1,trg_len): 


                

                


            output, (hidden, cell) = self.decoder(inp, (hidden, cell), context)
                


            

            

            outputs[t] = output 

            
             

                

                

                

                

                


            
            

            teacher_force = random.random() < teacher_forcing
            
            

            

            top1 = output.argmax(1) 

      
            
            

            

            inp = trg[:,t] if teacher_force else top1 

            


        
        return outputs
        
    
class Seq2SeqAttnLSTMPacked(nn.Module):      
    """
    encoder decoder are rnn/lstm networks
    usually we want hidden dims and layer on then to be the same
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    """    
    def __init__(self,encoder, decoder, params = {}, src_pad_idx=0):
        super().__init__() 

        self.encoder = encoder 
        self.decoder = decoder
        sel.src_pad_idx= src_pad_idx
        self.src_vocab_len = 0
        self.trg_vocab_len = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for p in params:




            if 'src_vocab_len' in p:
                self.src_vocab_len = params[p]
            if 'trg_vocab_len' in p:
                self.trg_vocab_len = params[p]    
            if 'device' in p:
                self.device = params[p]

        


        assert self.src_vocab_len> 0, "No src vocab len"
        assert self.trg_vocab_len> 0, "No trg vocab len"
        
    def create_mask(self, src): 

        mask = (src != self.src_pad_idx) 

        return mask
    
    
    def forward(self, src, src_len =None, trg = None, teacher_forcing= 0.5): 

        """
        src = [ batch size, src seq len]
        src_len = [batch size]
        trg = [ batch size, trg seq len,]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        """
        assert isinstance(src_len, torch.Tensor), "This is a seq2seq packed class, src len cannot be None"
        
        if trg == None: 

            trg = torch.zeros((src.shape[0], 25)).fill_(0).long().to(self.device)
            

            

            assert teacher_forcing == 0, "Must be zero during inference"
            
        

        trg_len = trg.shape[1] 

        bs = trg.shape[0] 

        


        outputs = torch.zeros(trg_len, bs, self.trg_vocab_len).to(self.device)

        encoder_outputs, (hidden,cell) = self.encoder(src, src_len) 
            


        context = encoder_outputs
        mask = self.create_mask(src) 



        inp = trg[:,0] 


        for t in range(1,trg_len): 

                
            output, (hidden, cell), _ = self.decoder(inp, (hidden, cell), context, mask) 

                

     
                

                



            

            

            outputs[t] = output 

            
             

                

                

                

                

                


            
            

            teacher_force = random.random() < teacher_forcing
            
            

            

            top1 = output.argmax(1) 

      
            
            

            

            inp = trg[:,t] if teacher_force else top1 

            


        
        return outputs

class Seq2SeqAttnGRU(nn.Module):      
    """
    encoder decoder are rnn/lstm networks
    usually we want hidden dims and layer on then to be the same
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    """    
    def __init__(self,encoder, decoder, params = {}, src_pad_idx=0):
        super().__init__() 

        self.encoder = encoder 
        self.decoder = decoder

        self.src_pad_idx= src_pad_idx

        self.src_vocab_len = 0
        self.trg_vocab_len = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
        for p in params:




            if 'src_vocab_len' in p:
                self.src_vocab_len = params[p]
            if 'trg_vocab_len' in p:
                self.trg_vocab_len = params[p]    
            if 'device' in p:
                self.device = params[p]

        


        assert self.src_vocab_len> 0, "No src vocab len"
        assert self.trg_vocab_len> 0, "No trg vocab len"

    
    def forward(self, src, trg = None, teacher_forcing= 0.5): 

        """
        src = [ batch size, src seq len]
        src_len = [batch size]
        trg = [ batch size, trg seq len,]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        """
        if trg == None: 

            trg = torch.zeros((src.shape[0], 25)).fill_(0).long().to(self.device)
            

            

            assert teacher_forcing == 0, "Must be zero during inference"
            
        

        trg_len = trg.shape[1] 

        bs = trg.shape[0] 

        


        outputs = torch.zeros(trg_len, bs, self.trg_vocab_len).to(self.device)


        encoder_outputs, hidden = self.encoder(src)
            


            
        context = encoder_outputs



        inp = trg[:,0] 

        
        for t in range(1,trg_len): 

            

                
            output, hidden,_ = self.decoder(inp, hidden, context) 

                

                
                

                




            

            

            outputs[t] = output 

            
             

                

                

                

                

                


            
            

            teacher_force = random.random() < teacher_forcing
            
            

            

            top1 = output.argmax(1) 

      
            
            

            

            inp = trg[:,t] if teacher_force else top1 

            


        
        return outputs    
    
class Seq2SeqAttnGRUPacked(nn.Module):      
    """
    encoder decoder are rnn/lstm networks
    usually we want hidden dims and layer on then to be the same
    https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
    """    
    def __init__(self,encoder, decoder, params = {}, src_pad_idx=0):
        
        super().__init__() 

        self.encoder = encoder 
        self.decoder = decoder

        self.src_pad_idx= src_pad_idx

        self.src_vocab_len = 0
        self.trg_vocab_len = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
        for p in params:




            if 'src_vocab_len' in p:
                self.src_vocab_len = params[p]
            if 'trg_vocab_len' in p:
                self.trg_vocab_len = params[p]    
            if 'device' in p:
                self.device = params[p]

        


        assert self.src_vocab_len> 0, "No src vocab len"
        assert self.trg_vocab_len> 0, "No trg vocab len"
        
    def create_mask(self, src): 

        mask = (src != self.src_pad_idx) 

        return mask
    
    
    def forward(self,src, src_len =None, trg = None, teacher_forcing= 0.5): 

        """
        src = [ batch size, src seq len]
        src_len = [batch size]
        trg = [ batch size, trg seq len,]
        teacher_forcing_ratio is probability to use teacher forcing
        e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        """
        assert isinstance(src_len, torch.Tensor), "This is a seq2seq packed class, src len cannot be None"

        if trg == None: 

            print ("This is Inference")
            trg = torch.zeros((src.shape[0], 25)).fill_(0).long().to(self.device)
            

            

            assert teacher_forcing == 0, "Must be zero during inference"
            
        

        trg_len = trg.shape[1] 

        bs = trg.shape[0] 

        


        outputs = torch.zeros(trg_len, bs, self.trg_vocab_len).to(self.device)


        encoder_outputs, hidden = self.encoder(src, src_len)
            


            
        context = encoder_outputs

        mask = self.create_mask(src) 




        inp = trg[:,0] 

        
        for t in range(1,trg_len): 

            

                
            output, hidden,_ = self.decoder(inp, hidden, context, mask) 

                

                
                

                




            

            

            outputs[t] = output 

            
             

                

                

                

                

                


            
            

            teacher_force = random.random() < teacher_forcing
            
            

            

            top1 = output.argmax(1) 

      
            
            

            

            inp = trg[:,t] if teacher_force else top1 

            


        
        return outputs
    
class MultiNet(nn.Module):  
    """
    this is for two model two input. 
    Change for three model three input and so on..
    """
    
    def __init__(self,net1, net2, net3):
        super().__init__() 

        self.model1 = net1
        self.model2 = net2
        self.model3 = net3
    """     
    def forward(self,x1,x2): 

        

        

 

        
        x1 = self.model1(x1)
 

       
        x2 = self.model2(x2)
 

        x = torch.cat((x1, x2), dim=1)
 

        x = self.model3(x)
 

 
        return x
    """
    def forward(self,x): 
        x1 = self.model1(x)
        
        x2= permuteTensor(0,2,1)(x) 

        x2 = self.model2(x2)
        
        x = torch.cat((x1, x2), dim=1) 

        x = self.model3(x)

 
        return x    

class SamplerSimilarLengthHFDataset(Sampler):
    """
    give batch size , drop last , shuffle and keyname of src in HF dataset. Dataset is assumed to be pytorch tensor
    """
    
    def __init__(self, dataset,batch_size, shuffle=True, drop_last= True, keyname= ''):
        
        assert keyname, "No keyname provided for dataset"
        self.keyname = keyname
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset 

        self.drop_last = drop_last
        
        self.get_indices()
        self.create_pooled_indices()
        self.create_batches()

    def get_indices(self):

        

        self.indices = [(i, s.shape[0]) for i, s in enumerate(self.dataset[self.keyname])] 

        

        

            
    def create_pooled_indices(self):
        if self.shuffle:
            random.shuffle(self.indices) 


        pooled_indices = []
        for i in range(0, len(self.indices), self.batch_size*100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size*100], key=lambda x: x[1]))
        

        self.pooled_indices = [x[0] for x in pooled_indices]
        

        
    def create_batches(self):
        
        self.batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]
        
        if self.drop_last:
            if len(self.dataset[self.keyname]) % self.batch_size == 0: 

                pass
            else:
                self.batches.pop()

        if self.shuffle:
            random.shuffle(self.batches)  
            
    def __iter__(self):
        for batch in self.batches:          
            yield batch
        
    
class BatchSamplerSimilarLength(Sampler):
    """
    dataloader gets fed the dataset - like train_data. The same dataset gets fed to this batchsampler along with batchsize and tokensizer
    to group inputs of similar seq lens.
    
    Returns indexes of dataset ( train_data for example) , where number of indexes = batchsize.
    The dataloader then uses these indexes and creates a batch of those entries from the dataset and sends to collate.
    
    """
    
    def __init__(self, dataset, batch_size,indices=None, shuffle=True, tokenizer = None, drop_last= True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.indices = indices
        self.tokenizer = tokenizer
        self.drop_last = drop_last
        
        self.get_indices()
        self.create_pooled_indices()
        self.create_batches()

    def get_indices(self):

        

        if self.indices is None:
            

            self.indices = [(i, len(self.tokenizer(s[0]))) for i, s in enumerate(self.dataset)] 

            

            

            
    def create_pooled_indices(self):
        
        if self.shuffle:
            random.shuffle(self.indices) 


        pooled_indices = []
        

        

        

        

        

        
        for i in range(0, len(self.indices), self.batch_size*100):
            pooled_indices.extend(sorted(self.indices[i:i + self.batch_size*100], key=lambda x: x[1]))


        
        

        

        self.pooled_indices = [x[0] for x in pooled_indices]


        
    def create_batches(self):
        
                

        self.batches = [self.pooled_indices[i:i + self.batch_size] for i in
                   range(0, len(self.pooled_indices), self.batch_size)]
        


        

        

        
        

        
        if self.drop_last:
            if len(self.dataset) % self.batch_size == 0: 

                pass
            else:
                self.batches.pop()

        if self.shuffle:
            random.shuffle(self.batches)
        
    def __iter__(self):

        for batch in self.batches:          
            yield batch 

            


            

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size

class Net(object):
    
    def __init__(self, logfile='/home/arnab/notebooks/ml/lregressions/runlogs',print_console =True,logtodisk=False):
        
        self.regress = False
        self.multiclass = False
        self.biclass = False
        self.savebest=False
        self.logger = log_setup('INFO',logtodisk,logfile)
        self.prntconsole=print_console
        self.chkptepoch = False
        self.savestart = False
        self.saveend =  False
        self.register_forward_callbacks = [] 

        self.register_backward_callbacks = []
       
            
    def setup_save(self, savedir = '', savestart = False, saveend = False):
        self.savestart = savestart
        self.saveend = saveend
        self.savedir =  savedir
        assert len(self.savedir) >0 , "No save directory entered...exiting"
        
    def saveModel(self, model = None,filename =''):
        if model:
            torch.save(model,self.savedir+filename+str(datetime.datetime.now())) 
        else:
            self.savedmodel ={}
            self.savedmodel['net'] = self.net.state_dict()
            self.savedmodel['opt'] = self.optimizer.state_dict()
            torch.save(self.savedmodel,self.savedir+filename+str(datetime.datetime.now()))
            
    def HFsavemodel(self,model,filedir=""):
        """
        model here needs to be a HF model , only then it has attribute of save_pretrained
        saves weights
        """
        assert filedir, "Enter file directory" 
        model.save_pretrained(filedir)
        
    def HFloadmodel(self,filedir=""):
        """
        mostly a example here
        loads with weights
        """
        model = AutoModelForMaskedLM.from_pretrained(filedir)
        

        return model
   
    def HFloadpretrainedmodel_from_dict(self, cfgdir='',dictdir =''):
        
        """
        THis is using the example where i loaded pretrained urlBERT.
        Note this does not load tokenizer
        
        """
        
        config_kwargs = {
                            "cache_dir": None,
                            "revision": 'main',
                            "use_auth_token": None,
                            "hidden_dropout_prob": 0.2,
                            "vocab_size": 5000,
                        }

        config = AutoConfig.from_pretrained("/content/drive/MyDrive/Colab Notebooks/My NN stuff/Modelsaves/urlBERTconfig.json",
                                            **config_kwargs)
  

        bert_model = AutoModelForMaskedLM.from_config(config=config)

 
        bert_model.resize_token_embeddings(config_kwargs["vocab_size"])
        


        bert_dict = torch.load("/content/drive/MyDrive/Colab Notebooks/My NN stuff/Modelsaves/urlBERT.pt", map_location='cpu')

        bert_model.load_state_dict(bert_dict)
        return bert_model
        
        
    def setupCuda(self):
        print ("Is GPU available ? : ", torch.cuda.is_available())  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        
    def configureNetwork(self,confobj =None, networktest = True, vae = None, RNN= False, rnnhc =False, conv1D = False, packed = True):
        """
        confobj is a dictionary of configuration class objects where confobj.cnnnet will carry cnn net cfg and other vars like confobj.fcin can be accessed
        confobj = {'CNN':[confobj1,confobj2],'FCN':[confobj2]}
        """
        if confobj and RNN:
            if rnnhc:
                self.net = RNNhc(confobj =confobj)
            elif packed:                
                

                self.net = RNNpacked(confobj =confobj)
            else:
                self.net = RNN_classification(confobj =confobj)
            
        elif confobj and conv1D:
            self.net = CNN1D(confobj =confobj)

        elif vae:
            self.net = VAE(encoder=vae['encoder'], decoder = vae['decoder'],fcin=vae['fcin'],zdims=vae['zdims'])
        else:
            self.net = ANN(confobj =confobj) 

            
        if networktest:
            if vae:
                self.network_test(VAE=True)
            else:
                self.network_test()
            

    def network_test(self, randinput = None, verbose = False, VAE=False, RNN = False):
        
        if randinput:
            tmpx = randinput
        else:
            tmpx, _ = next(iter(self.train_loader))


            if not RNN:
                tmpx = tmpx[:2]

        y = self.net(tmpx)
        print ("

        print (self.net)
        print ("

        print ('Shape of input', tmpx.detach().shape)
        if verbose:
            print ('Network input', tmpx.detach())
        if VAE:
            print ('Shape of output', y[3].detach().shape)
        else:
            print ('Shape of output', y.detach().shape)
        if verbose:
            if VAE:
                print ('Network output', y[3].detach())
            else:
                print ('Network output', y.detach())
        

        
    def memory_estimate(self):
        

        

        

        self.param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        
    
    
    def forwardcallbacks(self):
        if self.register_forward_callbacks:
            for x in self.register_forward_callbacks:
                x()
        else:
            return None
        
    def backwardcallbacks(self):
        if self.register_backward_callbacks:
            for x in self.register_backward_callbacks:
                x()
        else:
            return None        
        
    def setup_checkpoint(self,epoch=50,loss=None, acc=None, 
                         checkpointpath='/home/arnab/notebooks/ml/lregressions/checkpointsaves/'):
        
        """
        chkptepoch is interval to save checkpoints
        checkpointpath is the path
        crieteria  = chkptloss or chkptacc boolean value to decide
        three checkpoint dictionaries which overwrite checkpt1, checkpt2, checkpt3
        """
        
        self.chkptepoch =  epoch 

        self.checkpointpath = checkpointpath
        
        if loss:
            self.chkptloss = True
            self.chkptacc = False
            self.chkptlossval = loss
            
            self.checkpt1 = {'epoch': None,'net': None,'opt': None,'loss': loss +10}
            self.checkpt2 = {'epoch': None,'net': None,'opt': None,'loss': loss +10}
            self.checkpt3 = {'epoch': None,'net': None,'opt': None,'loss': loss +10}
        elif acc:
            self.chkptloss = False
            self.chkptacc = True
            self.chkptaccval = acc  
            self.checkpt1 = {'epoch': None,'net': None,'opt': None,'acc': acc -10}
            self.checkpt2 = {'epoch': None,'net': None,'opt': None,'acc': acc -10}
            self.checkpt3 = {'epoch': None,'net': None,'opt': None,'acc': acc -10}
        else:
            self.chkptloss = False
            self.chkptacc = False  
            self.chkptepoch =  False
        

    def loadModelfromdisk (self,modeldir='',optimizerdir=''):

        self.net.load_state_dict(torch.load(modeldir))
        optimizer.load_state_dict(torch.load(optimizerdir))
    
    def checkpoint(self,epoch=0):
        
        """
        Max of three based on loss/acc. min every epoch.If want just every epoch, then have loss or acc at values that def accepted.
        """
        if self.chkptloss:
            
            tmp = [self.checkpt1['loss'],self.checkpt2['loss'], self.checkpt3['loss']]
            maxis= tmp.index(max(tmp))
        
            if maxis == 0:

                if np.mean(self.batchLoss) < self.checkpt1['loss']:
                    self.checkpt1['loss'] = np.mean(self.batchLoss)
                    self.checkpt1['epoch'] = epoch
                    self.checkpt1['net'] = self.net.state_dict()
                    self.checkpt1['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt1,self.checkpointpath+"checkpt1"+str(datetime.datetime.now()))

            elif maxis == 1:

                if np.mean(self.batchLoss) < self.checkpt2['loss']:
                    self.checkpt2['loss'] = np.mean(self.batchLoss)
                    self.checkpt2['epoch'] = epoch
                    self.checkpt2['net'] = self.net.state_dict()
                    self.checkpt2['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt2,self.checkpointpath+"checkpt2"+str(datetime.datetime.now()))

            elif maxis == 2:

                if np.mean(self.batchLoss) < self.checkpt3['loss']:
                    self.checkpt3['loss'] = np.mean(self.batchLoss)
                    self.checkpt3['epoch'] = epoch
                    self.checkpt3['net'] = self.net.state_dict()
                    self.checkpt3['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt3,self.checkpointpath+"checkpt3"+str(datetime.datetime.now()))             
            
        elif self.chkptacc:
            
            tmp = [self.checkpt1['acc'],self.checkpt2['acc'], self.checkpt3['acc']]
            minis= tmp.index(min(tmp))
            
            if minis == 0:

                if np.mean(self.batchAcc) > self.checkpt1['acc']:
                    self.checkpt1['acc'] = np.mean(self.batchAcc)
                    self.checkpt1['epoch'] = epoch
                    self.checkpt1['net'] = self.net.state_dict()
                    self.checkpt1['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt1,self.checkpointpath+"checkpt1"+str(datetime.datetime.now()))

            elif minis == 1:

                if np.mean(self.batchAcc) > self.checkpt1['acc']:
                    self.checkpt1['acc'] = np.mean(self.batchAcc)
                    self.checkpt1['epoch'] = epoch
                    self.checkpt1['net'] = self.net.state_dict()
                    self.checkpt1['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt1,self.checkpointpath+"checkpt1"+str(datetime.datetime.now()))


            elif minis == 2:

                if np.mean(self.batchAcc) > self.checkpt1['acc']:
                    self.checkpt1['acc'] = np.mean(self.batchAcc)
                    self.checkpt1['epoch'] = epoch
                    self.checkpt1['net'] = self.net.state_dict()
                    self.checkpt1['opt'] = self.optimizer.state_dict()

                    torch.save(self.checkpt1,self.checkpointpath+"checkpt1"+str(datetime.datetime.now()))
            
        else:
            print("No checkpoint criteria selected\n")
            return None
                
    
    def configureTraining(self,epochs=500,lossfun=nn.CrossEntropyLoss(),optimizer='adam',lr=0.01, 
                          weight_decay=0,momentum=0.9, prntsummary = False, gpu= False):
        self.gpu = gpu
        self.lossfun = lossfun
        self.lr=lr
        self.epochs=epochs
        self.weight_decay=weight_decay 

        self.momentum=momentum 

        if optimizer =='sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        elif optimizer =='rmsprop':
            self.optimizer = torch.optim.RMSprop(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay,momentum=self.momentum)
        elif optimizer=='adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        if prntsummary:
            self.prnt_trainparams()
            
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        
        if self.gpu:
            print ("Is GPU available ? : ", torch.cuda.is_available())
            if torch.cuda.is_available():
                print ("

                print('ID of current CUDA device: ', torch.cuda.current_device())
                print('Name of current CUDA device is: ', torch.cuda.get_device_name(torch.cuda.current_device()))
                print('Amount of GPU memory allocated: ', torch.cuda.memory_allocated(torch.cuda.current_device()))
                print('Amount of GPU memory reserved: ', torch.cuda.memory_reserved(torch.cuda.current_device()))
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            else:
                self.gpu = False
        print("Processor device configured is: ", self.device)  

  
            
    def saveBest(self, accthreshold = 80.0, l1lossthresh=1.0, epochthreshold = 0, lossthreshold = 500, startlossval =  50000):
        
        self.savebest=True
        self.bestaccthresh = accthreshold
        self.bestl1thresh = l1lossthresh
        self.lossthreshold =  lossthreshold

        self.epochthresh = epochthreshold 

        self.bestTrain = {'acc':0.0, 'net':None, 'epoch':None,'opt':None,'loss':startlossval} 

        
              

    def dataIterables(self, sentences, targets, weights = {"train": 0.8, "valid": 0.1, "test": 0.1}):
        """
        sentences are a list of texts(reviews), targets are list of lists for one hot encode for multi labels or in case binary or multi class- then sholud be just a list of labels.
        in case of multilabel multiclass, the hot encoded values 0 and 1s are long ints
        """
        datapipe = []
        for i in range(len(sentences)):
            datapipe.append([sentences[i],targets[i]])
            
        N_ROWS = len(datapipe)


        self.datapipe = IterableWrapper(datapipe)
    
 




        

        

        self.train_dp, self.valid_dp, self.test_dp = self.datapipe.random_split(total_length=N_ROWS, 
                                                    weights=weights, seed = 0)  
        
        
    def dataIterables_to_list(self):
        
        self.train_dp_list = list(self.train_dp)
        self.valid_dp_list = list(self.valid_dp)
        self.test_dp_list = list(self.test_dp)

                          
    def makeVocabulary(self,dp= None, tokenizer = 'spacy', specials =  ["<UNK>", "<PAD>"], max_tokens =20000, default_index = 0,min_freq =2):
        
        

        
        def yield_tokens(data_iter):
            for text, _ in data_iter:
                yield self.tokenizer(text)
                
        if tokenizer:
            self.tokenizer = get_tokenizer(tokenizer)

        if dp:
             self.vocabulary = build_vocab_from_iterator(yield_tokens(dp), specials=specials,special_first= True, 
                                      max_tokens=max_tokens, min_freq = min_freq)
        else:
            self.vocabulary = build_vocab_from_iterator(yield_tokens(self.datapipe), specials=specials,special_first= True, 
                                      max_tokens=max_tokens,  min_freq = min_freq)
        self.vocabulary.set_default_index(default_index)
        self.PADDING_VALUE=self.vocabulary['<PAD>']

    def saveVocab(self, path):
        torch.save(self.vocabulary, path)
        
    def loadVocab(self, path):
        self.vocabulary = torch.load(path)
        
    def makeLoadersAdv(self, batch_size =64, collate_fn=None, iterables_to_list = False):
        
        """
        Batchsampler is called once for each loader, as it passed the entire dataset in a single call.
        So since we are creating train, valid and test loader, it is called three time in total
        
        NOTE : collate fn is not called at all during this makeloader function. Only batchsampler is called passing entire 
        datasets and that is what is taking long. Collate fun is only called when we iter the dataloader objects of self.train_loader
        and the other two. Means its using yield a single batch at a time to ram. So running a for loop over train_loader will call
        collate each time it loops and puts its on CPU ram, which then in traing code gets sent to gpu.
        So technically can send to GPU directly from collate - which may need to do when using more workers.
        """
        
        if iterables_to_list:
            self.dataIterables_to_list()       
        if collate_fn:
            self.collate_fn = collate_fn
        else:
            self.collate_fn =  self.collate_batch
        
        self.batch_size = batch_size
        
        self.train_loader = DataLoader(self.train_dp_list, 
                          batch_sampler=BatchSamplerSimilarLength(dataset = self.train_dp_list, 
                                                                  batch_size=self.batch_size, tokenizer = self.tokenizer),
                          collate_fn=self.collate_fn)

        self.valid_loader = DataLoader(self.valid_dp_list, 
                          batch_sampler=BatchSamplerSimilarLength(dataset = self.valid_dp_list, 
                                                                  batch_size=self.batch_size,
                                                                  shuffle=False, tokenizer = self.tokenizer),
                          collate_fn=self.collate_fn)
        
        self.test_loader = DataLoader(self.test_dp_list, 
                          batch_sampler=BatchSamplerSimilarLength(dataset = self.test_dp_list, 
                                                                  batch_size=self.batch_size,
                                                                  shuffle=False, tokenizer = self.tokenizer),
                          collate_fn=self.collate_fn)
        
        
    def makeLoaders(self, data,labels,train_size=.8,shuffle=True, batch_size=32,drop_last=True,testset=True):

    


        self.batch_size=batch_size

        self.train_data,self.test_data, self.train_labels,self.test_labels = \
                                  train_test_split(data, labels, train_size=train_size)
        if testset:
            train_data = torch.utils.data.TensorDataset(self.train_data,self.train_labels)
            test_data  = torch.utils.data.TensorDataset(self.test_data,self.test_labels)
        else:
            train_data = torch.utils.data.TensorDataset(torch.cat((self.train_data,self.test_data),0),torch.cat((self.train_labels,self.test_labels),0))
            test_data  = torch.utils.data.TensorDataset(self.test_data,self.test_labels)            

        self.train_loader = DataLoader(train_data,shuffle=shuffle,batch_size=self.batch_size,drop_last=drop_last)
        self.test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

        return self.train_loader, self.test_loader
    
    def collateseq2seqHFDatasetPack(self,batch):

        encoder_list, decoder_list = [], []
        x_lens, y_lens = [], []

        

        


        encoder_list = [example["src"] for example in batch] 

        x_lens = [example["src"].shape[0] for example in batch]
        decoder_list = [example["trg"] for example in batch] 

        y_lens = [example["trg"].shape[0] for example in batch]
        x_lens = torch.tensor(np.array(x_lens))
        y_lens = torch.tensor(np.array(y_lens))

        batchedsrc = pad_sequence(encoder_list, batch_first=True,padding_value=self.PADDING_VALUE)
        batchedtrg = pad_sequence(decoder_list, batch_first=True,padding_value=self.PADDING_VALUE)

        collated = {"src" : batchedsrc, "trg" : batchedtrg, "x_lens" : x_lens, "y_lens" : y_lens}

        return collated
    
    def collate_batch(self,batch):
        text_list, label_list = [], []
        for _text, _label in batch:
            processed_text = torch.tensor(self.vocabulary.lookup_indices(self.tokenizer(_text)))
            text_list.append(processed_text) 
            label_list.append(_label)
        


        label_torch = torch.tensor(np.array(label_list)).float()




            




    

    

        return pad_sequence(text_list, batch_first=True,padding_value=self.PADDING_VALUE), label_torch
        
        
    def embedvec_to_vocabtokn(self, vecmat,toknvec,vocabulary):   
        """vecmat = embeddings matrix
        tokenvec = vector for token
        give a embeded vector for a token , and get the token back
        """
        tokenls = []
        
        for i in torch.where((toknvec == vecmat).all(dim=1))[0].detach().numpy().tolist():
            tokenls.append(self.vocabulary.get_itos()[i])
            
        
        return tokenls
    
    def text_to_vocabidx(self, sentences):
        """
        here sentences is a list of numpy array of text
        """
        
        return [self.vocabulary.lookup_indices(self.tokenizer(i)) for i in sentences]
        
    
    def trainVAE(self,batchlogs=True,testset=True,verbose=False, lossreduction = False):
        """
        Note that self.net is returing a tuple and not just yhat
        """

        if batchlogs:
            self.logger.info("

            
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        if self.savestart:
            self.saveModel(filename='start')
            
            
        starttime = time.time()
        
        self.net.to(self.device) 


        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,_) in enumerate(self.train_loader):
                
                self.net.train()
                

                
                X = X.to(self.device) 


                z,z_mean,z_log_var,yHat = self.net(X)
                
                

                kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) 

                batchsize = kl_div.size(0) 



                kl_div = kl_div.mean()
    


                
                if lossreduction:
                    reconstructloss = self.lossfun(yHat,X)


                    self.loss = reconstructloss + kl_div


                else:               
                    pixelwise = self.lossfun(yHat,X)
    

                    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) 

    

                    pixelwise = pixelwise.mean() 

    

                    self.loss = pixelwise + kl_div
    


                


                    
                self.forwardcallbacks() 
                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
                self.backwardcallbacks()
                
                

                
                yHat = yHat.cpu()
                X = X.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  torch.mean(torch.abs(yHat.detach()-X.detach())).item()

                
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())

                if batchlogs:

                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                     "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))
                                     
                    if self.prntconsole:
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")

                                        


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)           
            
            if batchlogs:
                self.logger.info("

                                                                                                     
                if self.prntconsole:
                    
                    print("

                                                                                              tmpmeanbatchloss))

            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                
                
            

            """
            if testset:
                self.net.eval() 

                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) 

                    

                    

                    
                    X = X.to(self.device)
                    z,z_mean,z_log_var,predictions = self.net(X)
                    
                    predictions = predictions.cpu()
                    X = X.cpu()   
                    
                    tmptestloss = self.lossfun(predictions,X).item()
                    tmptestacc = torch.mean(torch.abs(predictions.detach()-X.detach())).item()

                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)

                    
                    if batchlogs: 
                        
                        self.logger.info("

                        if self.prntconsole:
                            print("

            """
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        if self.saveend:
            self.saveModel(filename='End')                           
            if self.savebest:          
                self.saveModel(model = self.bestTrain, filename='bestTrain')           
                           
        return self.trainAcc, self.testAcc,  self.losses,self.testloss                    
                    
    def trainVAEHFds(self,batchlogs=True,testset=True,verbose=False, Xkey = "hidden_state",lossreduction = False):
        """
        Note that self.net is returing a tuple and not just yhat
        """

        if batchlogs:
            self.logger.info("

            
        self.trainAcc = []
        self.reconstructionloss = []
        self.kldiv = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        if self.savestart:
            self.saveModel(filename='start')
            
            
        starttime = time.time()
        
        self.net.to(self.device) 


        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            self.batchkl = []
            self.reconstruct = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                

                
                X = batch[Xkey]
                X = X.to(self.device) 


                z,z_mean,z_log_var,yHat = self.net(X)
                
                

                kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1) 

                batchsize = kl_div.size(0) 



                kl_div = kl_div.mean()
    


                
                if lossreduction:
                    reconstructloss = self.lossfun(yHat,X)


                    self.loss = reconstructloss + kl_div


                else:               
                    pixelwise = self.lossfun(yHat,X)
    

                    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) 

    

                    pixelwise = pixelwise.mean() 

    

                    self.loss = pixelwise + kl_div
    


                


                    
                self.forwardcallbacks() 
                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
                self.backwardcallbacks()
                
                

                
                yHat = yHat.cpu()
                X = X.cpu()
                
                self.loss = self.loss.cpu()
                reconstructloss = reconstructloss.cpu()
                kl_div =  kl_div.cpu()
                
                
                tmpacc =  torch.mean(torch.abs(yHat.detach()-X.detach())).item()

                self.batchkl.append(kl_div.item()) 
                self.reconstruct.append(reconstructloss.item())
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())

                if batchlogs:

                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                     "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))
                    
                    self.logger.info("Reconstruction loss is %f and KL div is %f "% (reconstructloss,kl_div))
                                     
                    if self.prntconsole:
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")
                        
                        print ("Reconstruction loss is %f and KL div is %f "% (reconstructloss,kl_div))
                                        


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            self.reconstructionloss.append(np.mean(self.reconstruct))
            self.kldiv.append(np.mean(self.batchkl))
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)           
            
            if batchlogs:
                self.logger.info("

                                                                                                     
                if self.prntconsole:
                    
                    print("


            if testset:
                
                self.net.eval()
                with torch.no_grad():
                    
                    self.batchtestloss = []
                    self.batchtestAcc = []
                    
                    for batchidx, batch in enumerate(self.test_loader):
                        X = batch[Xkey]
                        X = X.to(self.device) 


                        z,z_mean,z_log_var,yHat = self.net(X)

                        

                        kl_div = -0.5 * torch.sum(1 + z_log_var 
                                              - z_mean**2 
                                              - torch.exp(z_log_var), 
                                              axis=1) 

                        batchsize = kl_div.size(0) 

                        kl_div = kl_div.mean()

                        if lossreduction:
                            reconstructloss = self.lossfun(yHat,X)
        

                            self.loss = reconstructloss + kl_div


                        else:               
                            pixelwise = self.lossfun(yHat,X)
            

                            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) 

            

                            pixelwise = pixelwise.mean() 

            

                            self.loss = pixelwise + kl_div
    


                        yHat = yHat.cpu()
                        X = X.cpu()
                        self.loss = self.loss.cpu()
                    
                        tmpacc =  torch.mean(torch.abs(yHat.detach()-X.detach())).item()

                
                        self.batchtestAcc.append(tmpacc)
                        self.batchtestloss.append(self.loss.item())

                        if batchlogs:

                            self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                             "TEST batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))

                            if self.prntconsole:
                                print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "TEST batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")

                                                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchtestAcc)
            tmpmeanbatchloss = np.mean(self.batchtestloss)

            self.testAcc.append(tmpmeanbatchacc)
            self.testloss.append(tmpmeanbatchloss)
                
                
                
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        if self.saveend:
            self.saveModel(filename='End')                           
            if self.savebest:          
                self.saveModel(model = self.bestTrain, filename='bestTrain')           
                           
        return self.trainAcc, self.testAcc,  self.losses, self.testloss, self.reconstructionloss, self.kldiv                    
                    


    
    def trainAE(self,batchlogs=True,testset=True,verbose=False):       

        if batchlogs:
            self.logger.info("

            
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        if self.savestart:
            self.saveModel(filename='start')

            
        starttime = time.time()
        
        self.net.to(self.device) 


        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,y) in enumerate(self.train_loader):
                
                self.net.train()
                

                
                X = X.to(self.device) 


                yHat = self.net(X)
                self.loss = self.lossfun(yHat,X)
                    
                self.forwardcallbacks() 
                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                
                self.backwardcallbacks()
                
                

                
                yHat = yHat.cpu()
                X = X.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  torch.mean(torch.abs(yHat.detach()-X.detach())).item()

                
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())

                if batchlogs:

                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                     "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))
                                     
                    if self.prntconsole:
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc abs error (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")

                                        


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)           
            
            if batchlogs:
                self.logger.info("

                                                                                                     
                if self.prntconsole:
                    
                    print("

                                                                                              tmpmeanbatchloss))

            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                
                
            

            if testset:
                self.net.eval() 

                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) 

                    

                    

                    
                    X = X.to(self.device)
                    predictions = self.net(X)
                    
                    predictions = predictions.cpu()
                    X = X.cpu()   
                    
                    tmptestloss = self.lossfun(predictions,X).item()
                    tmptestacc = torch.mean(torch.abs(predictions.detach()-X.detach())).item()

                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)

                    
                    if batchlogs: 
                        
                        self.logger.info("

                        if self.prntconsole:
                            print("

                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')        
        return self.trainAcc, self.testAcc,  self.losses,self.testloss 
    
    def trainseq2seqHFdatasetpacked(self,teacher_forcing = 1,clipping= 0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
    
        if batchlogs:
            self.logger.info("

        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 

        


        for epochi in tqdm.tqdm(range(self.epochs)):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                

                X = batch["src"]  
                y = batch["trg"]
                x_lens = batch["x_lens"]
                
                X = X.to(self.device) 

                y = y.to(self.device) 

                
                yHat = self.net(X,x_lens,y, teacher_forcing =  teacher_forcing)
                
                

                

                

                
                y = y.permute(1,0) 

                yHat_dim = yHat.shape[-1] 

                yHat = yHat[1:].view(-1, yHat_dim) 

                y = y[1:].flatten() 

                self.loss = self.lossfun(yHat,y)
                

                

                

                

                

                


                
                    
                self.forwardcallbacks() 

                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                

                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                

                

                

                
                
                

                
                

                

                

                

                

                

                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("

                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("

                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            

            if testset:
                
                self.net.eval() 

                with torch.no_grad():
                    





                    tmpbatchloss = [] 

                    tmpbatchacc = []
                    for batchidx, batch in enumerate(self.test_loader):
                        X = batch["src"]
                        y = batch["trg"]
                        x_lens = batch["x_lens"]

                        X = X.to(self.device) 

                        y = y.to(self.device)

                        pred = self.net(X,x_lens,y,teacher_forcing = 0) 



                        y = y.permute(1,0)
                        pred_dim = pred.shape[-1] 

                        pred = pred[1:].view(-1, pred_dim) 
                        y = y[1:].flatten() 


                        tmptestloss = self.lossfun(pred,y) 
                    
                        y = y.cpu().detach()
                        pred=pred.cpu().detach()
                        tmptestloss=tmptestloss.cpu()
                    
                        tmptestloss=tmptestloss.item()
                        predlabels = torch.argmax(pred,axis=1)
                        tmptestacc = 100*torch.mean((predlabels == y).float()).item()
                    
                        tmpbatchloss.append(tmptestloss)
                        tmpbatchacc.append( tmptestacc)
                    

                    

                    

                    tmpmeanbatchloss = np.mean(tmpbatchloss) 

                    tmpmeanbatchacc = np.mean(tmpbatchacc) 

                    
                    self.testloss.append(tmpmeanbatchloss) 

                    self.testAcc.append(tmpmeanbatchacc) 

                    
                    if batchlogs:

                        self.logger.info("

                        
                        if self.prntconsole:
        
                            print("

                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

          
        return self.trainAcc,self.testAcc, self.losses
        


    def trainseq2seqHFdataset(self,teacher_forcing = 1,clipping= 0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("

        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 

        


        for epochi in tqdm.tqdm(range(self.epochs)):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                

                X = batch["src"] 

                y = batch["trg"] 

                
                X = X.to(self.device) 

                y = y.to(self.device) 

                
                yHat = self.net(X,y, teacher_forcing =  teacher_forcing)
                
                

                

                

                
                y = y.permute(1,0) 

                yHat_dim = yHat.shape[-1] 

                yHat = yHat[1:].view(-1, yHat_dim) 

                y = y[1:].flatten() 

                self.loss = self.lossfun(yHat,y)
                

                

                

                

                

                


                
                    
                self.forwardcallbacks() 

                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                

                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                

                

                

                
                
                

                
                

                

                

                

                

                

                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("

                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("

                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            

            if testset:
                
                self.net.eval() 

                with torch.no_grad():
                    





                    tmpbatchloss = [] 

                    tmpbatchacc = []
                    for batchidx, batch in enumerate(self.test_loader):
                        X = batch["src"]
                        y = batch["trg"]

                        X = X.to(self.device) 

                        y = y.to(self.device)

                        pred = self.net(X,y,teacher_forcing = 0) 



                        y = y.permute(1,0)
                        pred_dim = pred.shape[-1] 

                        pred = pred[1:].view(-1, pred_dim) 
                        y = y[1:].flatten() 


                        tmptestloss = self.lossfun(pred,y) 
                    
                        y = y.cpu().detach()
                        pred=pred.cpu().detach()
                        tmptestloss=tmptestloss.cpu()
                    
                        tmptestloss=tmptestloss.item()
                        predlabels = torch.argmax(pred,axis=1)
                        tmptestacc = 100*torch.mean((predlabels == y).float()).item()
                    
                        tmpbatchloss.append(tmptestloss)
                        tmpbatchacc.append( tmptestacc)
                    

                    

                    

                    tmpmeanbatchloss = np.mean(tmpbatchloss) 

                    tmpmeanbatchacc = np.mean(tmpbatchacc) 

                    
                    self.testloss.append(tmpmeanbatchloss) 

                    self.testAcc.append(tmpmeanbatchacc) 

                    
                    if batchlogs:

                        self.logger.info("

                        
                        if self.prntconsole:
        
                            print("

                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

          
        return self.trainAcc,self.testAcc, self.losses
        


        
    def trainseq2seqSelfAttnHFdataset(self, clipping= 0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("

        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 

        


        for epochi in tqdm.tqdm(range(self.epochs)):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                
                

                X = batch["src"] 

                y = batch["trg"] 

                
                X = X.to(self.device) 

                y = y.to(self.device) 

                
                

                

                
                yHat, _ = self.net(X, y[:,:-1])
                 

                    
                

                

                
                yHat_dim = yHat.shape[-1] 

                
                yHat = yHat.contiguous().view(-1, yHat_dim) 

                y = y[:,1:].contiguous().view(-1) 

                

                self.loss = self.lossfun(yHat,y)
                

                

                

                

                

                


                    
                self.forwardcallbacks() 

                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                

                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                

                

                

                
                
                

                
                

                

                

                

                

                

                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("

                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("

                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            

            if testset:
                
                self.net.eval() 

                with torch.no_grad():
                    





                    tmpbatchloss = [] 

                    tmpbatchacc = []
                    for batchidx, batch in enumerate(self.test_loader):
                
                        X = batch["src"]
                        y = batch["trg"]

                        X = X.to(self.device) 

                        y = y.to(self.device)
                        
                        

                        pred,_ = self.net(X,y) 



                   
                        pred_dim = pred.shape[-1] 

                        pred = pred[1:].view(-1, pred_dim) 
                        y = y[1:].flatten() 


                        tmptestloss = self.lossfun(pred,y) 
                    
                        y = y.cpu().detach()
                        pred=pred.cpu().detach()
                        tmptestloss=tmptestloss.cpu()
                    
                        tmptestloss=tmptestloss.item()
                        predlabels = torch.argmax(pred,axis=1)
                        tmptestacc = 100*torch.mean((predlabels == y).float()).item()
                    
                        tmpbatchloss.append(tmptestloss)
                        tmpbatchacc.append( tmptestacc)
                    

                    

                    

                    tmpmeanbatchloss = np.mean(tmpbatchloss) 

                    tmpmeanbatchacc = np.mean(tmpbatchacc) 

                    
                    self.testloss.append(tmpmeanbatchloss) 

                    self.testAcc.append(tmpmeanbatchacc) 

                    
                    if batchlogs:

                        self.logger.info("

                        
                        if self.prntconsole:
        
                            print("

                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

          
        return self.trainAcc,self.testAcc, self.losses
        





    def trainseq2seq(self,teacher_forcing = 1,clipping= 0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("

        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 

        


        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,x_lens,y,y_lens) in enumerate(self.train_loader):
                
                self.net.train()
                

                
                X = X.to(self.device) 

                y = y.to(self.device) 

                
                yHat = self.net(X,x_lens,y, teacher_forcing =  teacher_forcing)
                
                

                

                

                
                y = y.permute(1,0) 

                yHat_dim = yHat.shape[-1] 

                yHat = yHat[1:].view(-1, yHat_dim) 

                y = y[1:].flatten() 

                self.loss = self.lossfun(yHat,y)
                

                

                

                

                

                


                
                    
                self.forwardcallbacks() 

                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                

                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                

                

                

                
                
                

                
                

                

                

                

                

                

                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("

                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("

                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            

            if testset:
                
                self.net.eval() 

                with torch.no_grad():
                    





                    tmpbatchloss = [] 

                    tmpbatchacc = []
                    for batchidx, (X,x_lens,y,y_lens) in enumerate(self.test_loader):

                        X = X.to(self.device) 

                        y = y.to(self.device)

                        pred = self.net(X,x_lens,y,teacher_forcing = 0) 



                        y = y.permute(1,0)
                        pred_dim = pred.shape[-1] 

                        pred = pred[1:].view(-1, pred_dim) 
                        y = y[1:].flatten() 


                        tmptestloss = self.lossfun(pred,y) 
                    
                        y = y.cpu().detach()
                        pred=pred.cpu().detach()
                        tmptestloss=tmptestloss.cpu()
                    
                        tmptestloss=tmptestloss.item()
                        predlabels = torch.argmax(pred,axis=1)
                        tmptestacc = 100*torch.mean((predlabels == y).float()).item()
                    
                        tmpbatchloss.append(tmptestloss)
                        tmpbatchacc.append( tmptestacc)
                    

                    

                    

                    tmpmeanbatchloss = np.mean(tmpbatchloss) 

                    tmpmeanbatchacc = np.mean(tmpbatchacc) 

                    
                    self.testloss.append(tmpmeanbatchloss) 

                    self.testAcc.append(tmpmeanbatchacc) 

                    
                    if batchlogs:

                        self.logger.info("

                        
                        if self.prntconsole:
        
                            print("

                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

          
        return self.trainAcc,self.testAcc, self.losses
        

        
        
        
    def trainDistilbertMask(clipping =0, Xkey='input_ids',attnkey = 'attention_mask', ykey='labels'):

        self.trainperplex = []
        self.losses   = []

        starttime = time.time()

        self.net.to(self.device) 


        


        for epochi in range(self.epochs):

            self.net.train() 


            batchAcc = []
            batchLoss = []

            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

            for batchidx, batch in enumerate(self.train_loader):

                self.net.train()
                


                X = batch[Xkey]  

                attn_mask = batch[attnkey] 

                y = batch[ykey] 

                X = X.to(self.device) 
                attn_mask = attn_mask.to(self.device)
                y = y.to(self.device) 

                outputs = net(X,attention_mask= attn_mask,labels = y)

                self.loss = outputs.loss 


                self.optimizer.zero_grad()

                self.loss.backward()

                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)

                self.optimizer.step()

                self.loss = loss.cpu()

                batchLoss.append(self.loss.item())

                print ('At Batchidx %d in epoch %d: '%(batchidx,epochi), "loss is %f "% (loss.item()))



                



            tmpmeanbatchloss = np.mean(batchLoss)
            try:
                perplexity = math.exp(tmpmeanbatchloss)
            except OverflowError:
                perplexity = float("inf")

            self.losses.append(tmpmeanbatchloss)
            self.trainperplex.append(perplexity)



            print("

                                                                                              tmpmeanbatchloss))


        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 


        return self.trainperplex, self.losses

        
        
        
    def trainTransformerFTmulticlass(self,clipping =0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False,
                                Xkey='input_ids',attnkey = 'attention_mask', ykey='label'):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("

        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 

        


        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                


                    
                X = batch[Xkey] 

                attn_mask = batch[attnkey] 
                y = batch[ykey] 

                
                X = X.to(self.device) 

                attn_mask = attn_mask.to(self.device)
                y = y.to(self.device) 

                
                yHat = self.net(X,attn_mask)
                
                self.loss = self.lossfun(yHat,y)
                    
                self.forwardcallbacks() 

                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                

                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                

                

                

                
                
                

                
                

                

                

                

                

                

                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("

                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("

                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            

            if testset:
                
                self.net.eval() 

                with torch.no_grad():
                    
                    batch = next(iter(self.test_loader)) 

                    

                    


                    
                    X = batch[Xkey] 

                    attn_mask = batch[attnkey] 
                    y = batch[ykey] 

                
                    X = X.to(self.device) 

                    attn_mask = attn_mask.to(self.device)
                    y = y.to(self.device) 

                

                    
                    pred = self.net(X,attn_mask)
                    
                    pred = pred.detach().cpu() 

                    y = y.cpu() 

                    
                    predlabels = torch.argmax( pred,axis=1 )
                    
                    tmptestloss = self.lossfun(pred,y.detach()).item()
                    tmptestacc = 100*torch.mean((predlabels == y.detach()).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append( tmptestacc)
                    

                    

                    

                    if batchlogs:

                        self.logger.info("

                        
                        if self.prntconsole:
        
                            print("

                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

        if testset:
            
            self.net.eval() 

            self.misclassifiedTest = np.where(predlabels != y.detach())[0]
    

            if prnt_misclassified:
                print ('

                if verbose :
                    print ("Rows in Testset: ",X[self.misclassifiedTest].detach())

        
        self.net.eval() 

        
        with torch.no_grad():
            batch = next(iter(self.train_loader))
            
            X = batch[Xkey] 

            attn_mask = batch[attnkey] 
            y = batch[ykey] 

            
            X = X.to(self.device) 

            attn_mask = attn_mask.to(self.device)
            y = y.to(self.device)
            
            pred = self.net(X,attn_mask)
            
            pred = pred.detach().cpu() 

            y = y.cpu() 

            
            predlabels = torch.argmax(pred,axis=1 )
            self.misclassifiedTrain = np.where(predlabels != y.detach())[0]
            if prnt_misclassified:
                print ('

                if verbose:
                    print("Rows in Trainset: ",X[self.misclassifiedTrain].detach())
         
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')   
                
        return self.trainAcc,self.testAcc, self.losses,  self.misclassifiedTrain, self.misclassifiedTest
        

        
        
        
        
    def trainmulticlassHFdataset(self,clipping =0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False,
                                Xkey  = 'X', ykey='y'):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("

        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 

        


        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                


                    
                X = batch[Xkey] 

                y = batch[ykey] 

                
                X = X.to(self.device) 

                y = y.to(self.device) 

                
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                    
                self.forwardcallbacks() 

                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                

                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                

                

                

                
                
                

                
                

                

                

                

                

                

                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("

                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("

                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            

            if testset:
                
                self.net.eval() 

                with torch.no_grad():
                    
                    batch = next(iter(self.test_loader)) 

                    

                    

                    
                    X = batch[Xkey] 

                    y = batch[ykey] 

                    
                    X = X.to(self.device) 

                    y = y.to(self.device)
                    
                    pred = self.net(X)
                    
                    pred = pred.detach().cpu() 

                    y = y.cpu() 

                    
                    predlabels = torch.argmax( pred,axis=1 )
                    
                    tmptestloss = self.lossfun(pred,y.detach()).item()
                    tmptestacc = 100*torch.mean((predlabels == y.detach()).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append( tmptestacc)
                    

                    

                    

                    if batchlogs:

                        self.logger.info("

                        
                        if self.prntconsole:
        
                            print("

                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

        if testset:
            
            self.net.eval() 

            self.misclassifiedTest = np.where(predlabels != y.detach())[0]
    

            if prnt_misclassified:
                print ('

                if verbose :
                    print ("Rows in Testset: ",X[self.misclassifiedTest].detach())

        
        self.net.eval() 

        
        with torch.no_grad():
            batch = next(iter(self.train_loader))
            
            X = batch[Xkey] 

            y = batch[ykey] 

            
            X = X.to(self.device) 

            y = y.to(self.device)
            
            pred = self.net(X)
            
            pred = pred.detach().cpu() 

            y = y.cpu() 

            
            predlabels = torch.argmax(pred,axis=1 )
            self.misclassifiedTrain = np.where(predlabels != y.detach())[0]
            if prnt_misclassified:
                print ('

                if verbose:
                    print("Rows in Trainset: ",X[self.misclassifiedTrain].detach())
         
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')   
                
        return self.trainAcc,self.testAcc, self.losses,  self.misclassifiedTrain, self.misclassifiedTest
        



        
    def trainmulticlass(self,clipping =0, batchlogs=True, prnt_misclassified=True, testset=True,verbose=False):
   
        self.multiclass = True
        if batchlogs:
            self.logger.info("

        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 

        


        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,y) in enumerate(self.train_loader):
                
                self.net.train()
                

                
                X = X.to(self.device) 

                y = y.to(self.device) 

                
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                

                

                    
                self.forwardcallbacks() 

                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()
                
                tmpacc =  100*torch.mean((torch.argmax(yHat.detach(),axis=1) == y.detach()).float()).item()                
                
                self.batchAcc.append(tmpacc) 
                

                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                

                

                

                
                
                

                
                

                

                

                

                

                

                
                if batchlogs:

                    
                    
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                               "batchacc is %f and loss is %f "% (tmpacc,
                                                                   self.loss.item()))
                           


                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)            

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                                                                    
                self.logger.info("

                                                                                                     tmpmeanbatchloss))
                if self.prntconsole:
                                                                      
                    print("

                                                                                              tmpmeanbatchloss))
                                                                                          
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
                    
                

            

            if testset:
                
                self.net.eval() 

                with torch.no_grad():
                    
                    X,y = next(iter(self.test_loader)) 

                    

                    

                    
                    X = X.to(self.device) 

                    y = y.to(self.device)
                    
                    pred = self.net(X)
                    
                    pred = pred.detach().cpu() 

                    y = y.cpu() 

                    
                    predlabels = torch.argmax( pred,axis=1 )
                    
                    tmptestloss = self.lossfun(pred,y.detach()).item()
                    tmptestacc = 100*torch.mean((predlabels == y.detach()).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append( tmptestacc)
                    

                    

                    

                    if batchlogs:

                        self.logger.info("

                        
                        if self.prntconsole:
        
                            print("

                     
        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

        if testset:
            
            self.net.eval() 

            self.misclassifiedTest = np.where(predlabels != y.detach())[0]
    

            if prnt_misclassified:
                print ('

                if verbose :
                    print ("Rows in Testset: ",X[self.misclassifiedTest].detach())

        
        self.net.eval() 

        with torch.no_grad():
            X,y = next(iter(self.train_loader))
            
            X = X.to(self.device) 

            y = y.to(self.device)
            
            pred = self.net(X)
            
            pred = pred.detach().cpu() 

            y = y.cpu() 

            
            predlabels = torch.argmax(pred,axis=1 )
            self.misclassifiedTrain = np.where(predlabels != y.detach())[0]
            if prnt_misclassified:
                print ('

                if verbose:
                    print("Rows in Trainset: ",X[self.misclassifiedTrain].detach())
         
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')            
        return self.trainAcc,self.testAcc, self.losses,  self.misclassifiedTrain, self.misclassifiedTest
        

        
        
    def trainTransformerFTbiclass(self,clipping = 0,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False,
                                 Xkey='input_ids',attnkey = 'attention_mask', ykey='label'):

        self.biclass = True
        if batchlogs:
            self.logger.info("

        
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 
        
        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, batch in enumerate(self.train_loader):
                
                self.net.train()
                

                
                X = batch[Xkey] 

                attn_mask = batch[attnkey] 
                y = batch[ykey] 

                
                X = X.to(self.device) 

                attn_mask = attn_mask.to(self.device)
                y = y.to(self.device) 

                                
              
                yHat = self.net(X,attn_mask)
                
                self.loss = self.lossfun(yHat,y)
                
                self.forwardcallbacks()
                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)       
                
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()                
           
                tmpacc = 100*torch.mean(((yHat.detach() > 0 ) == y.detach()).float()).item() 

            

            
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                
                

                

                

                

                
                if batchlogs:
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:
                        
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc is %f and loss is %f "% (tmpacc,
                                                         self.loss.item()))

                


            

            

            
            tmpmeanbatchloss = np.mean(self.batchLoss)
            tmpmeanbatchacc = np.mean(self.batchAcc)
            
            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                self.logger.info("

                                                                                                    tmpmeanbatchloss))
                if self.prntconsole:
                    
                    print("

                                                                                              tmpmeanbatchloss))
 
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())



            

            if testset:
                self.net.eval() 

                with torch.no_grad():
                    self.batchtestloss = []
                    self.batchtestAcc = []
                    
                    for batchidx, batch in enumerate(self.test_loader):


                    

                    

                    
                        X = batch[Xkey] 

                        attn_mask = batch[attnkey] 
                        y = batch[ykey] 


                        X = X.to(self.device) 

                        attn_mask = attn_mask.to(self.device)
                        y = y.to(self.device) 


                        predlabels = self.net(X,attn_mask)

                        predlabels = predlabels.detach().cpu() 

                        y = y.detach().cpu() 


                        tmptestloss = self.lossfun(predlabels,y).item()
                        tmptestacc = 100*torch.mean(((predlabels > 0 ) == y).float()).item()
                    
                        self.batchtestloss.append(tmptestloss)
                        self.batchtestAcc.append(tmptestacc)
                        
                        if batchlogs:

                            self.logger.info("

                            if self.prntconsole:                          
                                print("

                            
                    tmpmeanbatchloss = np.mean(self.batchtestloss)
                    tmpmeanbatchacc = np.mean(self.batchtestAcc)
            
                    self.testAcc.append(tmpmeanbatchacc)
                    self.testloss.append(tmpmeanbatchloss)
                    
                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

        if testset:
            self.net.eval() 

            self.misclassifiedTest = np.where((predlabels > 0 ) != y.detach())[0]

    

            if prnt_misclassified:
                print ('

                if verbose:
                    print("Misclassified Rows in Testset: ",X[self.misclassifiedTest].detach()) 

                   

        self.net.eval() 

        with torch.no_grad():
            
            batch = next(iter(self.train_loader))
            
            X = batch[Xkey] 

            attn_mask = batch[attnkey] 
            y = batch[ykey] 

            
            X = X.to(self.device) 

            attn_mask = attn_mask.to(self.device)
            y = y.to(self.device)           
            
            predlabels = self.net(X,attn_mask)
            
            predlabels = predlabels.detach().cpu() 

            y = y.cpu() 

            
            self.misclassifiedTrain = np.where((predlabels > 0 ) != y.detach())[0]
            if prnt_misclassified:
                print ('

                if verbose:
                    print ("Misclassified Rows in Trainset: ",X[self.misclassifiedTrain].detach())
            
            
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')
                
        return self.trainAcc,self.testAcc, self.losses, self.misclassifiedTrain, self.misclassifiedTest
                
        

    def trainbinaryclassHFDataset(self,clipping = 0,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False,
                                 Xkey ='hidden_state', ykey='label'):

        self.biclass = True
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 
        
        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx,batch in enumerate(self.train_loader):
                
                self.net.train()
                

                
                X = batch[Xkey] 

                y = batch[ykey] 


                X = X.to(self.device) 

                y = y.to(self.device) 

                                
              
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                
                self.forwardcallbacks()
                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)                
                
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()                
           
                tmpacc = 100*torch.mean(((yHat.detach() > 0 ) == y.detach()).float()).item() 

            

            
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                
                

                

                

                

                
                if batchlogs:
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:
                        
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc is %f and loss is %f "% (tmpacc,
                                                         self.loss.item()))

                


            

            

            
            tmpmeanbatchloss = np.mean(self.batchLoss)
            tmpmeanbatchacc = np.mean(self.batchAcc)
            
            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                self.logger.info("

                                                                                                    tmpmeanbatchloss))
                if self.prntconsole:
                    
                    print("

                                                                                              tmpmeanbatchloss))
 
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())



            

            if testset:
                self.net.eval() 

                with torch.no_grad():
                    
                    batch = next(iter(self.test_loader)) 

                    

                    

                    
                    X = batch[Xkey] 

                    y = batch[ykey] 

                    X = X.to(self.device) 

                    y = y.to(self.device)                    
                    
                    predlabels = self.net(X)
                    
                    predlabels = predlabels.detach().cpu() 

                    y = y.detach().cpu() 

                    
                    tmptestloss = self.lossfun(predlabels,y).item()
                    tmptestacc = 100*torch.mean(((predlabels > 0 ) == y).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)
                    if batchlogs:
                        
                        self.logger.info("

                        if self.prntconsole:                          
                            print("

                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

        if testset:
            self.net.eval() 

            self.misclassifiedTest = np.where((predlabels > 0 ) != y.detach())[0]

    

            if prnt_misclassified:
                print ('

                if verbose:
                    print("Misclassified Rows in Testset: ",X[self.misclassifiedTest].detach()) 

                   

        self.net.eval() 

        with torch.no_grad():
            batch = next(iter(self.train_loader))
            X = batch[Xkey] 

            y = batch[ykey] 

            
            X = X.to(self.device) 

            y = y.to(self.device)            
            
            predlabels = self.net(X)
            
            predlabels = predlabels.detach().cpu() 

            y = y.cpu() 

            
            self.misclassifiedTrain = np.where((predlabels > 0 ) != y.detach())[0]
            if prnt_misclassified:
                print ('

                if verbose:
                    print ("Misclassified Rows in Trainset: ",X[self.misclassifiedTrain].detach())
            
            
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')
                
        return self.trainAcc,self.testAcc, self.losses, self.misclassifiedTrain, self.misclassifiedTest




    def trainbinaryclass(self,clipping = 0,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False):

        self.biclass = True
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []
        
        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)
        
        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()
        
        self.net.to(self.device) 
        
        for epochi in range(self.epochs):
            
            self.net.train() 

            
            self.batchAcc = []
            self.batchLoss = []
            
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)
            
            for batchidx, (X,y) in enumerate(self.train_loader):
                
                self.net.train()
                

                
                X = X.to(self.device) 

                y = y.to(self.device) 

                                
              
                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                

                self.forwardcallbacks()
                
                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)                
                
                self.optimizer.step()
                
                self.backwardcallbacks()

                

                
                

                
                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()                
           
                tmpacc = 100*torch.mean(((yHat.detach() > 0 ) == y.detach()).float()).item() 

            

            
                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())
                
                

                

                

                

                

                
                

                

                

                

                
                if batchlogs:
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                    "batchacc is %f and loss is %f "% (tmpacc,
                                                                      self.loss.item()))
                    if self.prntconsole:
                        
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc is %f and loss is %f "% (tmpacc,
                                                         self.loss.item()))

                


            

            

            
            tmpmeanbatchloss = np.mean(self.batchLoss)
            tmpmeanbatchacc = np.mean(self.batchAcc)
            
            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)
            
            if batchlogs:
                self.logger.info("

                                                                                                    tmpmeanbatchloss))
                if self.prntconsole:
                    
                    print("

                                                                                              tmpmeanbatchloss))
 
            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())



            

            if testset:
                self.net.eval() 

                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) 

                    

                    

                    
                    X = X.to(self.device) 

                    y = y.to(self.device)                    
                    
                    predlabels = self.net(X)
                    
                    predlabels = predlabels.detach().cpu() 

                    y = y.detach().cpu() 

                    
                    tmptestloss = self.lossfun(predlabels,y).item()
                    tmptestacc = 100*torch.mean(((predlabels > 0 ) == y).float()).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)
                    if batchlogs:
                        
                        self.logger.info("

                        if self.prntconsole:                          
                            print("

                        
        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

        if testset:
            self.net.eval() 

            self.misclassifiedTest = np.where((predlabels > 0 ) != y.detach())[0]

    

            if prnt_misclassified:
                print ('

                if verbose:
                    print("Misclassified Rows in Testset: ",X[self.misclassifiedTest].detach()) 

                   

        self.net.eval() 

        with torch.no_grad():
            X,y = next(iter(self.train_loader))
            
            X = X.to(self.device) 

            y = y.to(self.device)            
            
            predlabels = self.net(X)
            
            predlabels = predlabels.detach().cpu() 

            y = y.cpu() 

            
            self.misclassifiedTrain = np.where((predlabels > 0 ) != y.detach())[0]
            if prnt_misclassified:
                print ('

                if verbose:
                    print ("Misclassified Rows in Trainset: ",X[self.misclassifiedTrain].detach())
            
            
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')
                
        return self.trainAcc,self.testAcc, self.losses, self.misclassifiedTrain, self.misclassifiedTest





    def trainmultilabel(self,clipping = 0,numclasses = 3,batchlogs=True,prnt_misclassified=True, testset=True,verbose=False):

        self.multilabel = True
        self.trainAcc = []
        self.testAcc  = []
        self.losses   = []
        self.testloss = []

        onestensor = torch.ones(numclasses).detach()

        self.misclassifiedTrain= np.array(None)
        self.misclassifiedTest= np.array(None)

        if self.savestart:
            self.saveModel(filename='start')

        starttime = time.time()

        self.net.to(self.device)

        for epochi in range(self.epochs):

            self.net.train() 


            self.batchAcc = []
            self.batchLoss = []

            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

            for batchidx, (X,y) in enumerate(self.train_loader):

                self.net.train()
                
                X = X.to(self.device) 

                y = y.to(self.device) 



                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)

                self.forwardcallbacks()

                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)                
                
                self.optimizer.step()

                self.backwardcallbacks()



                


                


                yHat = yHat.cpu()
                y = y.cpu()
                self.loss = self.loss.cpu()   

                tmpacc = 100*(
                    torch.where(
                        (((yHat.detach() > 0 ) == y.detach()) == onestensor).all(dim=1))[0].shape[0])/y.detach().shape[0]

                self.batchAcc.append(tmpacc)
                self.batchLoss.append(self.loss.item())

                

                

                

                

                

                


                if batchlogs:
                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                        "batchacc is %f and loss is %f "% (tmpacc,
                                                                          self.loss.item()))
                    if self.prntconsole:

                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                "batchacc is %f and loss is %f "% (tmpacc,
                                                             self.loss.item()))

                


            

            


            tmpmeanbatchloss = np.mean(self.batchLoss)
            tmpmeanbatchacc = np.mean(self.batchAcc)

            self.trainAcc.append(tmpmeanbatchacc)
            self.losses.append(tmpmeanbatchloss)

            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)

            if batchlogs:
                self.logger.info("

                                                                                                    tmpmeanbatchloss))
                if self.prntconsole:

                    print("

                                                                                              tmpmeanbatchloss))

            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):
    

                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                   
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())
    


            

            if testset:
                self.net.eval() 

                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) 

                    

                    


                    X = X.to(self.device) 

                    y = y.to(self.device)                    

                    predlabels = self.net(X)
                    tmptestloss = self.lossfun(predlabels,y)
                    
                    

                    
                    predlabels = predlabels.detach().cpu() 

                    y = y.detach().cpu() 

                    
                    tmptestloss = tmptestloss.cpu().item()
                    tmptestacc = 100*(torch.where((((predlabels > 0 ) == y) == onestensor).all(dim=1))[0].shape[0])/y.shape[0]
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)

                    if batchlogs:

                        self.logger.info("

                        if self.prntconsole:                          
                            print("



        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) 

        

        if testset:
            self.net.eval() 

            self.misclassifiedTest = torch.where((((predlabels > 0 ) == y) == onestensor).all(dim=1))[0]
            print ('

            if verbose:
                print("Misclassified Rows in Testset: ",X[self.misclassifiedTest].detach()) 


        self.net.eval() 

        with torch.no_grad():
            X,y = next(iter(self.train_loader))

            X = X.to(self.device) 

            y = y.to(self.device)            

            predlabels = self.net(X)

            predlabels = predlabels.detach().cpu() 

            y = y.detach().cpu() 


            self.misclassifiedTrain = torch.where((((predlabels > 0 ) == y) == onestensor).all(dim=1))[0]
            if prnt_misclassified:
                print ('

                if verbose:
                    print ("Misclassified Rows in Trainset: ",X[self.misclassifiedTrain].detach())


        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')

        return self.trainAcc,self.testAcc, self.losses, self.misclassifiedTrain, self.misclassifiedTest


    
    
    def trainregression(self,clipping =0,batchlogs=True, testset=True):
        
        self.regress = True

        self.trainAcc = []
        self.testAcc  = []
        





        self.losses   = []
        self.testloss = [] 


        
        if self.savestart:
            self.saveModel(filename='start')

            
        starttime = time.time()
        
        self.net.to(self.device) 

            
        if batchlogs:
            self.logger.info("


        for epochi in range(self.epochs):
            
            self.net.train() 

            self.batchAcc = []


            self.batchLoss = []
    
            print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

            for batchidx, (X,y) in enumerate(self.train_loader):
                
                self.net.train()
                
                X = X.to(self.device) 

                

                yHat = self.net(X)
                self.loss = self.lossfun(yHat,y)
                
                self.forwardcallbacks()

                

                self.optimizer.zero_grad()
                self.loss.backward()
                
                if clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), clipping)
                    
                self.optimizer.step()
                
                self.backwardcallbacks()
                
                yHat = yHat.cpu()
                X = X.cpu()
                self.loss = self.loss.cpu()

                

                

                

                tmpacc = torch.mean(torch.abs(yHat.detach()-y.detach())).item()
                self.batchAcc.append(tmpacc)


                self.batchLoss.append(self.loss.item())

                if batchlogs:

                    self.logger.info('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                                     "batchacc L1loss (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()))
                                     
                    if self.prntconsole:
                        print ('At Batchidx %d in epoch %d: '%(batchidx,epochi),
                            "batchacc L1loss (mean if batches) is %f and loss is %f "% (tmpacc,self.loss.item()),"\n")

                


            

            

            
            tmpmeanbatchacc = np.mean(self.batchAcc)
            tmpmeanbatchloss = np.mean(self.batchLoss)

            self.trainAcc.append(tmpmeanbatchacc)


            self.losses.append(tmpmeanbatchloss)
            
            if self.chkptepoch and epochi !=0 and epochi%self.chkptepoch == 0 :
                self.checkpoint(epoch = epochi)           
            
            if batchlogs:
                self.logger.info("

                                                                                                     
                if self.prntconsole:
                    
                    print("

                                                                                              tmpmeanbatchloss))

            if self.savebest and epochi > self.epochthresh: 

                if (tmpmeanbatchloss < self.lossthreshold) and (tmpmeanbatchloss < self.bestTrain['loss']):


                    self.bestTrain['acc'] = tmpmeanbatchacc
                    self.bestTrain['epoch'] = epochi                  
                    self.bestTrain['net'] = copy.deepcopy( self.net.state_dict())
                    self.bestTrain['loss'] = tmpmeanbatchloss
                    self.bestTrain['opt'] = copy.deepcopy( self.optimizer.state_dict())                    
                
                
            

            if testset:
                self.net.eval() 

                with torch.no_grad():
                    X,y = next(iter(self.test_loader)) 

                    

                    

                    X = X.to(self.device)
                    predictions = self.net(X)
                    
                    predictions = predictions.cpu()
                    X = X.cpu()
                    
                    tmptestloss = self.lossfun(predictions,y).item()
                    tmptestacc = torch.mean(torch.abs(predictions.detach()-y.detach())).item()
                    
                    self.testloss.append(tmptestloss)
                    self.testAcc.append(tmptestacc)


                    if batchlogs:
                        
                        self.logger.info("

                        if self.prntconsole:
                            print("

                        
        if self.saveend:
            self.saveModel(filename='End')
            if self.savebest:
                self.saveModel(model = self.bestTrain, filename='bestTrain')
        return self.trainAcc, self.testAcc, self.losses,self.testloss
                           

    def prnt_trainparams(self,skip_lastbatchsize=True,verbose=False):
        
        print ("

        print (self.net)
        print ("\n")




























        if verbose :
            
            print ("

            for X,y in self.train_loader:
                print(X.detach()[0],y.detach()[0])
                break
























        print("

        print ("The model has " +str(sum(p.numel() for p in self.net.parameters() if p.requires_grad))+" trainable params")
        
        print("

        print("Loss function: ",self.lossfun,"\n", "Learning Rate: ", self.lr,"\n","Epochs :", self.epochs,"\n","Optimizer :", self.optimizer)
                           
        
    def lossplot(self, corrcoeff=False):
        
        

        
        

        figloss = go.Figure()
        figloss.add_trace(go.Scatter(x=[i for i in range(len(self.losses))], y=self.losses,name="Training Loss"))
        
        figloss.add_trace(go.Scatter(x=[i for i in range(len(self.testloss))], y=self.testloss,name="Test Loss"))
            
        figloss.update_layout(title="Loss Curve" , xaxis_title="Epochs", 
                          yaxis_title="Loss", height=400, width=500)
        figloss.show()
        
        figacc = go.Figure()
                           
        figacc.add_trace(go.Scatter(x=[i for i in range(len(self.trainAcc))], y=self.trainAcc, name="Training Accuracy"))
        figacc.add_trace(go.Scatter(x=[i for i in range(len(self.testAcc))], y=self.testAcc, name="Test Accuracy"))
        
        figacc.update_layout(title="Accuracy", xaxis_title="Epochs", yaxis_title="Training  & Test Accuracy",
                             height=400, width=500)
        figacc.show()
        
        if corrcoeff:
            figcorr = go.Figure()
            figcorr.add_trace(go.Scatter(x=[i for i in range(len(self.trainCorr))], y=self.trainCorr, name="Training Correlation"))
            figcorr.add_trace(go.Scatter(x=[i for i in range(len(self.testCorr))], y=self.testCorr, name="Test Correlation"))
            
            figcorr.update_layout(title="Correlation-Coefficient", xaxis_title="Epochs",
                                  yaxis_title="Training  & Test Corrcoeff",height=400, width=500)
   
        if corrcoeff:
            self.net.eval() 

            with torch.no_grad():
                
                yhattrain = self.net(self.train_data) 

                yhattest = self.net(self.test_data) 

                
            figpred = go.Figure()
 
            

            corrTrain = np.corrcoef(yhattrain.detach().T,self.train_labels.T)[1,0]
            corrTest  = np.corrcoef(yhattest.detach().T, self.test_labels.T)[1,0]


            figpred.add_trace(go.Scatter(x=torch.flatten(yhattrain.detach()).numpy(), y=torch.flatten(self.train_labels.detach()).numpy(),name="Training predictions r= "+str(corrTrain))) 

            figpred.add_trace(go.Scatter(x=torch.flatten(yhattest.detach()).numpy(), y=torch.flatten(self.test_labels.detach()).numpy(),name="Test predictions r="+str(corrTest)))
            
            figpred.update_layout(title="Training and Test predictions vs actual readings (all data) ", xaxis_title="Predicted value", 
                          yaxis_title="True value", height=600, width=1000)
            figpred.show()
    
    
    def accuracyPerclass(self, misclass=None ,y=None):
    
        if self.biclass:

            print ("Number of missclassification for label 1 :",(y[misclass==1]==1).sum())
            print ("Accuracy in classifying label 1 :", ((y==1).sum().item() - (y[misclass==1]==1).sum().item()) / ((y==1).sum().item()))
            print ("Number of missclassification for label 0 :",(y[misclass==1]==0).sum())
            print ("Accuracy in classifying label 0 :", ((y==0).sum().item() - (y[misclass==1]==0).sum().item()) / ((y==0).sum().item()))

        elif multi:
            pass

    def aprf(self, y=None,yhat=None):

        

        









        
        self.sprfdict= {'accuracy':0,'precision':0,'recall':0,'f1':0 }

        if self.biclass:


            self.sprfdict['accuracy']  = skm.accuracy_score (y,yhat)
            self.sprfdict['precision'] = skm.precision_score(y,yhat)
            self.sprfdict['recall']    = skm.recall_score(y,yhat)
            self.sprfdict['f1']        = skm.f1_score(y,yhat)
            return self.sprfdict

        elif self.multiclass:

            

            

            


            self.sprfdict['accuracy']  = skm.accuracy_score (y,yhat)
            self.sprfdict['precision'] = skm.precision_score(y,yhat,average = 'weighted')
            self.sprfdict['recall']    = skm.recall_score(y,yhat,average = 'weighted')
            self.sprfdict['f1']        = skm.f1_score(y,yhat,average = 'weighted')
            return self.sprfdict


    def confusionmatrix(self,y=None, yhat=None,labels =[],heatmap = False):
        conf = skm.confusion_matrix(y,yhat)
        conf.astype(int)
        if heatmap and labels:
            
            fig = px.imshow(conf,  aspect="auto", x= labels, y = labels)
            fig.show()
        return conf
                    
                