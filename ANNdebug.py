import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import seaborn as sns
import numpy as np
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




    
def debugFCNweights(net, concat=True, perlayer=False):
    
    """
    returns dicttionary of weights for each linear layer
    """
    
    perlayerw={}
    concatallweights = np.array([])
    
    if concat:
        
        for i in net.layers:
            
            if 'Linear' in str(type(net.layers[i])):
                
                concatallweights = np.concatenate((concatallweights, net.layers[i].weight.detach().flatten().numpy()))
             
                
    if perlayer:

        for c,i in enumerate(net.layers):
            
            if 'Linear' in str(type(net.layers[i])):
                
                perlayerw[c] = net.layers[i].weight.detach().flatten().numpy()
    
    return concatallweights, perlayerw


def getperlayerFCNweights(net):
    
    perlayerw={}
    
    for c,i in enumerate(net.layers):
        
            
        if 'Linear' in str(type(net.layers[i])):

            perlayerw[c] = net.layers[i].weight.detach().numpy()
    
    return perlayerw


def getFCNconcatweights(net):
    
    """
    returns numpy martrix of weights from  each linear layer
    """
    
    concatallweights = np.array([])

    for i in net.layers:

        if 'Linear' in str(type(net.layers[i])):

            concatallweights = np.concatenate((concatallweights, net.layers[i].weight.detach().flatten().numpy()))
    return concatallweights


def getperlayerFCNgrads(net):
    
    perlayerg={}
    
    for c,i in enumerate(net.layers):
        
            
        if 'Linear' in str(type(net.layers[i])):

            perlayerg[c] = net.layers[i].weight.grad.detach().numpy()
    
    return perlayerg    

def getFCNconcatgrads(net):
    
    """
    returns numpy martrix of weights from  each linear layer
    """
    
    concatallgrads = np.array([])

    for i in net.layers:

        if 'Linear' in str(type(net.layers[i])):

            concatallgrads = np.concatenate((concatallgrads, net.layers[i].weight.grad.detach().flatten().numpy()))
    return concatallgrads
    
    
def FCNWeightsHist(net, concatenated=True,perlayer=False):
    
    if concatenated:
        
        concatallweights = np.array([])
        
        for i in net.layers:
            
            if 'Linear' in str(type(net.layers[i])):
                
                concatallweights = np.concatenate((concatallweights, net.layers[i].weight.detach().flatten().numpy()))
                

        fig = px.histogram(concatallweights,title ='Concatenated Weights',nbins=1)
        fig.show()
        
    if perlayer:
        
        for i in net.layers:
            
            if 'Linear' in str(type(net.layers[i])):
                
                w= net.layers[i].weight.detach().flatten().numpy()
                
                fig = px.histogram(w,title ='Weights for Layer '+str(i),nbins=1)
                fig.show() 
                             
                    
def CNNparamaterStats(net,prntweights=False, VAE = False):
    
    

    
    totalparams = 0
    
    if VAE:
        
        for i in net.encoder:
            print("Params in encoder Layer "+str(i), sum(param.numel() for param in net.encoder[i].parameters()))
            totalparams =  totalparams + sum(param.numel() for param in net.encoder[i].parameters())
        for i in net.decoder:
            print("Params in decoder Layer "+str(i), sum(param.numel() for param in net.decoder[i].parameters()))
            totalparams =  totalparams + sum(param.numel() for param in net.decoder[i].parameters())     
        print ("Total params in all Encoder and Decoder Layers: ", totalparams)
            
    else:
        
        for i in net.layers:


    


    


            if 'conv' in str(type(net.layers[i])):

                print("Params in Conv Layer "+str(i), sum(param.numel() for param in net.layers[i].parameters()))
                totalparams =  totalparams + sum(param.numel() for param in net.layers[i].parameters()) 

            elif 'pool' in str(type(net.layers[i])):
                print("Params in Pool Layer "+str(i), sum(param.numel() for param in net.layers[i].parameters()))
                totalparams =  totalparams + sum(param.numel() for param in net.layers[i].parameters()) 
                    
        print ("Total params in all Convolution Layers: ", totalparams)
        
        
                                             
def FCNparameterStats(net,onlylinear=True,prntweights=False,verbose =False, VAE =  False):
  
 

    
    biasc=0
    weightc=0
    nparams=0
    numnodes=0
    totalparams = 0
    
    if VAE:
        
        totalparams =  totalparams + sum(param.numel() for param in net.z_mean.parameters()) 
        totalparams =  totalparams + sum(param.numel() for param in net.z_log_var.parameters())
        print ("Total params in reparameterization FCN Layers: ", totalparams)
        
        
    else:
        
        print ("


        for i in net.layers:

            if 'Linear' or 'BatchNorm1d'  in str(type(net.layers[i])):

                nparams = nparams+ np.sum([ p.numel() for p in net.layers[i].parameters() if p.requires_grad ])

                if verbose:

                    print("Layer: ", type(net.layers[i]), "\n")
                    print ("weight shape: ",net.layers[i].weight.detach().shape)
                    print ("bias shape: ", net.layers[i].bias.detach().shape)
                    print ("\n")

                    for n,v in net.layers[i].named_parameters():
                        if 'bias' in n:
                            print ("Number of units in layer "+str(i)+": ", len(v))
                            numnodes= numnodes + len(v)


                    for name,vec in net.layers[i].named_parameters():
                        if prntweights:
                            print ("Name: ", name)
                            print ("Matrix: ",vec.detach(),"\n")

                        if 'bias' in name:
                            biasc+=vec.numel()

                        if 'weight' in name:
                            weightc+=vec.numel()

        print ('Total traininable paramerters in FCN: ', nparams)

        if verbose:

            print ("\n")
            print ("Number of weights paramters in FCN", weightc, "\n")
            print ("Number of bias paramters in FCN", biasc, "\n")
            print ("Total number of Units in FCN: ", numnodes,"\n")
    
    
  
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
              




def hook_prnt_activations(name=''):
    def hook(model, input, output):
        print ("Activation output size for Layer "+name+ ": ",output.detach().size())
        print ("Activation output for Layer "+name+ ": ",output.detach())
    return hook

activation = {}
inputs = {}

def hook_return_activations(name=''):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def hook_prnt_activation_norms(name=''):
    def hook(model, input, output):
        print ("Activation output norm for Layer "+name+ ": ",output.detach().norm())
    return hook

def hook_prnt_inputs(name='', data = False):
    def hook(model, input, output):
        print ("Size of input stack to layer "+name+ ": ", len(input))
        for c,i in enumerate(input):
            print ("Size of "+str(c) + "th element of input stack to layer "+name+ ": ", i.detach().size())
            if data:
                print (str(c) + "th element of input stack to layer "+name+ ": ",i.detach())
    return hook

def hook_return_inputs(name=''):
    def hook(model, input, output):
        inputs[name] = []
        for i in input:
            inputs[name].append(i.detach())
    return hook

def hook_prnt_inputs_norms(name=''):
    def hook(model, input, output):
        print ("Size of input stack to layer "+name+ ": ", len(input))
        for c,i in enumerate(input):
            print ("Size of "+str(c) + "th element of input stack to layer "+name+ ": ", i.detach().size())
            print ("Norm is: ", i.detach().norm())
    return hook

def hook_prnt_activations_stats(name=''):
    
    def hook(model, input, output):
        print ("Activation output size for Layer "+name+ ": ",output.detach().size())
        print ("Max, Min, Mean and Var for this Activation output : ",torch.max(output.detach()),torch.min(output.detach()),torch.mean(output.detach()), torch.var(output.detach()))
        
    return hook
               
def hook_prnt_inputs_stats(name=''):
    def hook(model, input, output):
        print ("Size of input stack to layer "+name+ ": ", len(input))
        for c,i in enumerate(input):
            print ("Size of "+str(c) + "th element of input stack to layer "+name+ ": ", i.detach().size())
            print ("Max, Min, Mean and Var for this input : ",torch.max(i.detach()),torch.min(i.detach()),torch.mean(i.detach()), torch.var(i.detach()))
    return hook               
    


def hook_prnt_weights_grad_stats(name=''):
    def hook(grad):
        print ("Max, Min, Mean and Var of gradients in tensor "+name+ ": ",torch.max(grad.detach()),torch.min(grad.detach()),torch.mean(grad.detach()), torch.var(grad.detach()))
        print ("Norm of gradients in tensor "+name+ ": ",grad.detach().norm())
    return hook 

def callback_prnt_allgrads_stats(net, VAE =False, decoder = False, encoder =False):
    
    def hook():
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        concatallgrads = torch.empty([0]).detach()


        concatallgrads.to(device)
    
        if VAE:
            
            try:
                concatallgrads = torch.cat((concatallgrads, net.z_mean.weight.grad.detach().flatten()))
                concatallgrads = torch.cat((concatallgrads, net.z_log_var.weight.grad.detach().flatten()))
            except AttributeError:
                pass
            
            if decoder:
                for i in net.decoder:

                    try:
                        concatallgrads = torch.cat((concatallgrads, net.decoder[i].weight.grad.detach().flatten()))
                    except AttributeError:
                        pass
            if encoder:
                for i in net.encoder:

                    try:
                        concatallgrads = torch.cat((concatallgrads, net.encoder[i].weight.grad.detach().flatten()))
                    except AttributeError:
                        pass                
            
        else:    

            for i in net.layers:

    


    

    


    

    

    

    

                try:    
                    concatallgrads = torch.cat((concatallgrads, net.layers[i].weight.grad.detach().flatten()))
                except AttributeError:
                    pass

        print ("Max, Min, Mean and Var of gradients: ",concatallgrads.max(), concatallgrads.min(), concatallgrads.mean(), concatallgrads.var())
        print ("Norm of gradients: ",concatallgrads.norm())
        
    return hook


def callback_prnt_weights_stats(net,modname):
    """
    net here is specific instance of layer in net for which weights stats are been checked
    for example conv layer might be net = gnet.net.layers['1'][0]
    modname is name for printing purposes 
    """
    

    
    def hook():
        numpyweight = net.weight.detach().flatten()
        print ("Max, Min, Mean and Var of weights for layer "+modname+ ": ",numpyweight.max(), numpyweight.min(), numpyweight.mean(), numpyweight.var())
        print ("Norm of weights for layer "+modname+ ": ",numpyweight.norm())
        
    return hook

def callback_prnt_allweights_stats(net, VAE =False, decoder = False, encoder =False):
    
    def hook():
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



        concatallweights = torch.empty([0]).detach()
        
        concatallweights.to(device)
        
        if VAE:
            
            try:
                concatallweights = torch.cat((concatallweights, net.z_mean.weight.detach().flatten()))
                concatallweights = torch.cat((concatallweights, net.z_log_var.weight.detach().flatten()))
            except AttributeError:
                pass
            
            if decoder:
                for i in net.decoder:

                    try:
                        concatallweights = torch.cat((concatallweights, net.decoder[i].weight.detach().flatten()))
                    except AttributeError:
                        pass
            if encoder:
                for i in net.encoder:

                    try:
                        concatallweights = torch.cat((concatallweights, net.encoder[i].weight.detach().flatten()))
                    except AttributeError:
                        pass                
            
            
            
        else:
            
            for i in net.layers:

    


    

    


    

    

    

                try:
                    concatallweights = torch.cat((concatallweights, net.layers[i].weight.detach().flatten()))
                except AttributeError:
                    pass

        print ("Max, Min, Mean and Var of weights: ",concatallweights.max(), concatallweights.min(), concatallweights.mean(), concatallweights.var())
        print ("Norm of weights: ",concatallweights.norm())

    return hook












