# -*- coding: utf-8 -*-
"""CSIC 2010 Distilbert HTTP2VEC
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

"""**Load CSIC 2010 dataset**

- Original dataset from http://www.isi.csic.es/dataset/ and I have converted to huggingface dataset available here :


https://huggingface.co/datasets/bridge4/CSIC2010_dataset_domain_adaptation

https://huggingface.co/datasets/bridge4/CSIC2010_dataset_classification


"""

def load_pickle_todf(filename=''):
    assert filename,"No filename provided"
    df = pd.read_pickle(filename)
    return df

dfdomainrequests = load_pickle_todf(filename='trainrequestsdomainadapt')
domaintrain_fullreq = Dataset.from_pandas(dfdomainrequests)

classifiertraining = load_pickle_todf(filename='classifierset_norm_anoma')
classifiertraining = Dataset.from_pandas(classifiertraining)

classifiertraining = classifiertraining.class_encode_column('label')
train_test = classifiertraining.train_test_split(test_size=0.3,seed =2018, stratify_by_column="label")

dataset = datasets.DatasetDict({
    'domain_adaptation': domaintrain_fullreq,
    'train': train_test['train'],
    'test': train_test['test']})


### HTTP headers in the original dataset are seperated by \\n, replace these with one whitespace

def replacecarriagereturn(example):

    example["requests"] =  example["requests"].replace("\\n"," ")
    return example

dataset['domain_adaptation']  = dataset['domain_adaptation'].map(replacecarriagereturn)
dataset['train']  = dataset['train'].map(replacecarriagereturn)
dataset['test']  = dataset['test'].map(replacecarriagereturn)

"""**Load distilbert to continue pre-training**

- Using legitimate HTTP requests from CSIC 2010 dataset, we continue pre-training for masked language modeling and domain adapt the model as an HTTP2vec for better HTTP embeddings.
"""

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

"""**Tokenize dataset for domain adaptation**"""

def tokenize_function(examples):

    result = tokenizer(examples["requests"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_datasets = dataset['domain_adaptation'].map(tokenize_function, batched=True, batch_size=1000)
tokenized_datasets= tokenized_datasets.remove_columns("requests")

"""**Concat and chunk data for masked language modeling**"""

chunk_size = 128
def concat_chunk(examples):

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])


    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size

    # Split concatanated examples by chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }

    # Create a new labels column for mask training
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(concat_chunk, batched=True, batch_size=1000)

"""**Whole word masking function to use as collator**"""

wwm_probability = 0.15


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)

"""**Dataloader and training function**"""

batch_size = 64
epochs = 2
train_loader = DataLoader(lm_datasets,shuffle=True,batch_size=batch_size, collate_fn=whole_word_masking_data_collator)
optimizer = AdamW(model.parameters(), lr=0.00005)

print ("Is GPU available ? : ", torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def trainDistelbertMask(net, optimizer, device, epochs, train_loader,
                        clipping =0, Xkey='input_ids',attnkey = 'attention_mask', ykey='labels'):

    trainperplex = []
    losses   = []

    starttime = time.time()

    net.to(device)

    for epochi in range(epochs):

        net.train()

        batchAcc = []
        batchLoss = []

        print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime)

        for batchidx, batch in enumerate(train_loader):

            net.train()


            X = batch[Xkey]

            attn_mask = batch[attnkey]

            y = batch[ykey]

            X = X.to(device)
            attn_mask = attn_mask.to(device)
            y = y.to(device)

            outputs = net(X,attention_mask= attn_mask,labels = y)

            loss = outputs.loss

            optimizer.zero_grad()

            loss.backward()

            if clipping > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping)

            optimizer.step()

            loss = loss.cpu()

            batchLoss.append(loss.item())

            print ('At Batchidx %d in epoch %d: '%(batchidx,epochi), "loss is %f "% (loss.item()))



            ##### end of batch loop######


        tmpmeanbatchloss = np.mean(batchLoss)
        try:
            perplexity = math.exp(tmpmeanbatchloss)
        except OverflowError:
            perplexity = float("inf")

        losses.append(tmpmeanbatchloss)
        trainperplex.append(perplexity)



        print("##Epoch %d averaged batch training perplexity is %f and loss is %f"%(epochi,perplexity,
                                                                                          tmpmeanbatchloss))


    print ("Time ( in secs) elapsed since start of training : ",time.time()-starttime) # final time once all training done

    return trainperplex, losses


trainperplex, losses = trainDistelbertMask(model, optimizer, device, epochs, train_loader)
model.save_pretrained("distbert_request_domain")
