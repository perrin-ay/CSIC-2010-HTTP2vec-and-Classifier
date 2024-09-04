# -*- coding: utf-8 -*-
"""CSIC 2010 Classification using trained Distilbert
"""

import datasets
from datasets import load_dataset,Value, Sequence, Features
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoModelForMaskedLM
from transformers import default_data_collator
from transformers import DataCollatorForLanguageModeling
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoConfig, AutoModelForMaskedLM, BertForMaskedLM
from transformers import pipeline
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import time,sys,os
import math
import collections
import itertools
import random

from ANNhelper import set_all_seeds,log_setup,configuration, ANN,VAE, Net, custReshape, Trim, CNN1D, RNN_classification
from ANNhelper import BatchSamplerSimilarLength, hiddenBidirectional, hiddenUnidirectional, squeezeOnes
from ANNhelper import BidirectionextractHiddenfinal, UnidirectionextractHiddenfinal, permuteTensor, globalMaxpool, MultiNet
from ANNhelper import LSTMhc,UnidirectionextractHiddenCell,UnidirectionalextractOutput,Linearhc,RNNhc,unsqueezeOnes
from ANNhelper import concatTwotensors, concatThreetensors, UnidirectionextractHidden, decoder_cho, hcHiddenonlyBidirectional
from ANNhelper import hcBidirectional, BidirectionextractHCfinal, Bidirectionfullprocess, activationhc, Linearhchiddencell
from ANNhelper import Attention, decoderGRU_attn_bahdanau, decoderGRU_cho, Seq2SeqAttnGRU, decoder_attn_bahdanau
from ANNhelper import GRULinearhchidden, activationh, standin,UnpackpackedOutputHidden, GRUBidirectionfullprocess
from ANNhelper import SamplerSimilarLengthHFDataset, Seq2SeqAttnGRUPacked, Seq2SeqAttnLSTMPacked, Seq2SeqLSTMPacked
from AttentionHelper import Seq2SeqSelfAttn, DecoderLayer, DecoderSelfAttn, PositionwiseFeedforwardLayer
from AttentionHelper import MultiHeadAttentionLayer, EncoderLayer, EncoderSelfAttn, TransformerFTseqclassifier

from ANNdebug import CNNparamaterStats,FCNparameterStats, hook_prnt_activations, hook_prnt_activation_norms
from ANNdebug import hook_prnt_inputs, hook_prnt_weights_grad_stats, callback_prnt_allweights_stats
from ANNdebug import callback_prnt_allgrads_stats, callback_prnt_weights_stats
from ANNdebug import hook_prnt_inputs_stats, hook_prnt_activations_stats, hook_prnt_inputs_norms, hook_return_activations, hook_return_inputs
from ANNdebug import activation as hookactivations
from ANNdebug import inputs as hookinputs

"""**Load domain adapted http2vec distilbert**

"""

from transformers import DistilBertTokenizer, DistilBertModel
modelclass = DistilBertModel.from_pretrained("distbert_request_domain")
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

"""**Load CSIC 2010 data for classification training**"""

def load_pickle_todf(filename=''):
    assert filename,"No filename provided"
    df = pd.read_pickle(filename)
    return df

dfdomainrequests = load_pickle_todf(filename='trainrequestsdomainadapt')
domaintrain_fullreq = Dataset.from_pandas(dfdomainrequests)

classifiertraining = load_pickle_todf(filename='classifierset_norm_anoma')
classifiertraining = Dataset.from_pandas(classifiertraining)

print (domaintrain_fullreq,classifiertraining )
classifiertraining = classifiertraining.class_encode_column('label')
train_test = classifiertraining.train_test_split(test_size=0.3,seed =2018, stratify_by_column="label")


dataset = datasets.DatasetDict({
    'domain_adaptation': domaintrain_fullreq,
    'train': train_test['train'],
    'test': train_test['test']})

### the headers are seperated by \\n, instead replace these with one whitespace

def replacecarriagereturn(example):

    example["requests"] =  example["requests"].replace("\\n"," ")
    return example

dataset['domain_adaptation']  = dataset['domain_adaptation'].map(replacecarriagereturn)
dataset['train']  = dataset['train'].map(replacecarriagereturn)
dataset['test']  = dataset['test'].map(replacecarriagereturn)

traintestdataset = datasets.DatasetDict({"train":dataset["train"], "test":dataset['test']})

def tokenize(batch):
    return tokenizer(batch["requests"], padding=False, truncation=True)


traintestdataset_encoded = traintestdataset.map(tokenize, batched=True, batch_size=1000)
traintestdataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

#### Casting label as pytorch float for training



feats = traintestdataset_encoded['train'].features.copy()
feats['label'] = Value('float32')
traintestdataset_encoded['train']=traintestdataset_encoded['train'].cast(feats)



featstest = traintestdataset_encoded['test'].features.copy()
featstest['label'] = Value('float32')
traintestdataset_encoded['test']=traintestdataset_encoded['test'].cast(featstest)

"""**Distilbert classifier**

- using cls pooling to get embeddings for HTTP requests from domain adapted http2vec distilbert
- embeddings fed to classifier head for binary classification training
- End to end training of domain adapted distilbert and classifier head for binary classification 
"""

class TransformerFTseqclassifier(nn.Module):

    def __init__(self , transformermodel, device, num_labels= 1):

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
        x= x.squeeze() # for binary class as x is [batchsize, 1] and labels is [batchsize]
        return x

"""**Setup training params**"""

distilbertclassifier  = Net()
distilbertclassifier.setupCuda()
device = distilbertclassifier.device
distilbertclassifier.net = TransformerFTseqclassifier(modelclass, device,num_labels=1)
distilbertclassifier.net.to(device)
distilbertclassifier.optimizer = torch.optim.Adam(distilbertclassifier.net.parameters(),
                                                  lr=0.00002)
distilbertclassifier.gpu = True
distilbertclassifier.lossfun = nn.BCEWithLogitsLoss()
distilbertclassifier.epochs=3

"""**Setup dataloader with collate function of whole word masking**"""

batch_size = 16
drop_last=True
shuffle = True


train_loader = DataLoader(traintestdataset_encoded['train'], shuffle=shuffle, batch_size=batch_size, drop_last=drop_last,
                         collate_fn =  DataCollatorWithPadding(tokenizer=tokenizer, padding= True, max_length = 512))
                          # model_max_length

test_loader =  DataLoader(traintestdataset_encoded['test'], shuffle=shuffle,
                          batch_size=batch_size,
                          collate_fn =  DataCollatorWithPadding(tokenizer=tokenizer, padding= True, max_length = 512))

distilbertclassifier.train_loader, distilbertclassifier.test_loader =  train_loader, test_loader

"""**Training**"""

result = distilbertclassifier.trainTransformerFTbiclass(clipping =1, Xkey ='input_ids',attnkey = 'attention_mask',
                                                   ykey='labels')
distilbertclassifier.savedir ="distbert_request_classify"
distilbertclassifier.saveModel()
