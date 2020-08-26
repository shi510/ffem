# FFEM  
FFEM stands for Face Feature Embedding Module.  
It supports triplet-loss and its variants.  
It also supports RFW dataset.  
It is tested on tensorflow-v2.3.0.  

## Prepare RFW Dataset
You have RFW dataset folder structure as below.  
```
RFW  
  |-- BUPT-Balancedface  
  |         |-- images/race_per_7000  
  |         |         |-- African  
  |         |         |-- Asian  
  |         |         |-- Caucasian  
  |         |         |-- Indian  
  |-- RFW  
  |         |-- images/test/data  
  |         |         |-- African  
  |         |         |-- Asian  
  |         |         |-- Caucasian  
  |         |         |-- Indian  
  |         |-- images/test/txts  
  |         |         |-- African  
  |         |         |-- Asian  
  |         |         |-- Caucasian  
  |         |         |-- Indian  
```

## Things You Should Know Before Training Your Model
It is difficult to train an embedding model with triplet loss from scratch.  
It often fails to converge and results in f(x)=0, where f(x) is an embeddings.  
To remedy this, there are a few options.  
```
1. Select triplet pair carefully with large mini-batch (>= 1800).  
2. Pretrain an embedding model as a classifier with softmax-cross-entropy loss.  
3. Try to train with other metric losses.  
```
In this project, it uses second option.  
Pretrain first and fine-tune the pretrained model with metric losses.  

## How To Train Your Face Embedding Model
```
1. Modify configuration settings in example/train/config.py.  
2. Execute commands below.  
  export PYTHONPATH=$(pwd)  
  python example/train/main.py  
```