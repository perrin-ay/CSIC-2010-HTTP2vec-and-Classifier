# CSIC-2010-HTTP2vec-and-Classifier

---

## Hierarchical design

#### Domain adaptation of distilBERT to create a HTTP2vec model

- Masked language modeling using Huggingface
- Using legitimate HTTP requests from CSIC 2010 dataset, we continue pre-training for masked language modeling and domain adapt the model as an HTTP2vec for better HTTP embeddings.
  
**Input**
```
[CLS] get / tienda1 / index. jsp http / 1. 1 user - agent :  Mozilla / 5. 0 ( compatible ; konqueror / 3. 5 ; linux ) khtml / 3. 5. 8 ( like gecko ) pragma : no - cache cache - control : no - cache accept : text / xml, application / xml, application / xhtml + xml, text / html ; q = 0. 9, text / plain ; q = 0. 8, image / png , * / * ; q = 0. 5 accept - encoding : x - g'
```
**Target**
```
[CLS] get / tienda1 / index. jsp http / 1. 1 user - agent : [MASK] [MASK] [MASK] / 5. 0 ( compatible ; konqueror / 3. 5 ; linux ) khtml / 3. 5. 8 ( like gecko ) pragma : no - cache cache - control [MASK] no [MASK] [MASK] accept : text / xml, application [MASK] xml, application / xhtml + xml, text / html ; q = 0. 9, text [MASK] plain ; [MASK] = 0. 8, image / png [MASK] * / * ; q = 0. 5 accept - encoding : x - g'
```

- Domain adapted model can be found here: https://huggingface.co/bridge4/distilbert_HTTPtoVec_CSIC2010
- Dataset used can be found here: https://huggingface.co/datasets/bridge4/CSIC2010_dataset_domain_adaptation

![image](https://github.com/user-attachments/assets/5169252e-6356-490d-8a64-bed1fccd4efc)

#### Classification 

- CSIC 2010 consists of legitimate and attack HTTP requests
- Create a network with domain adapted distBERT HTTP2vec model with a classifier head of FCN
- CLS pooling : In the forward pass use CLS embeddings of each HTTP request and pass to FCN for supervised binary classification.
- In backprop update weight end to end of the network ( distBERT and FCN)
- Trained model : https://huggingface.co/bridge4/CSIC2010_webATTACK_transformer_classifier
- Dataset: https://huggingface.co/datasets/bridge4/CSIC2010_dataset_classification

![image](https://github.com/user-attachments/assets/99585a38-7b4e-4b4c-9f8b-6b86fa4272b0)

---

## Results

- Full training logs : https://github.com/perrin-ay/CSIC-2010-HTTP2vec-and-Classifier/blob/main/full%20training%20logs.txt
- 
```
{'Training Accuracy': 99.45662227155013, 'Testing Accuracy': 99.59016393442623 , 'Loss': 0.022516824751084765}
```
