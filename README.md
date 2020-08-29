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
It often fails to converge and results in f(x)=0, where f(x) is an embedding vector.  
To remedy this, there are a few options.  
```
1. Select triplet pair carefully with large mini-batch (>= 1800).
2. Pretrain an embedding model as a classifier with softmax-cross-entropy loss.
3. Try to train with other metric losses.
```
In this project, it uses second option.  
Pretrain first and fine-tune the pretrained model with metric losses.  

## Common Settings
1. export PYTHONPATH=$(pwd)  
2. 'train_path' and 'test_path' in configuration file.  

## First Step (Classifier)
1. Choose 'batch_size' and 'num_identity' in configuration file for consideration of memory capacity.  
2. Set 'train_classifier' to `True` in configureation file.  
3. Run -> python example/train/main.py  

## Last Step (Metric Learning)
1. Set 'train_classifier' to `False` in configureation file.  
2. Set 'num_identity' to `None` in configuration file, It does not affects memory usage when 'train_classifier' option is `False`.  
3. Choose 'metric_loss' option in configureation file.  
4. Run -> python example/train/main.py  

## References
1. [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)
2. [Deep Face Recognition, VGGFACE](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
3. [RFW Face Dataset](http://www.whdeng.cn/RFW/index.html)
4. https://github.com/davidsandberg/facenet/
5. https://github.com/omoindrot/tensorflow-triplet-loss
