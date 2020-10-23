# FFEM  
FFEM stands for Face Feature Embedding Module.  
This project is tested on tensorflow-v2.3.0.  
The tensorflow-addons is needed for tfa.images.  

## Things You Should Know Before Training Your Model
It is difficult to train an embedding model with a triplet loss from scratch.  
It often fails to converge and results in f(x)=0, where f(x) is an embedding vector.  
To remedy this, there are a few options.  
```
1. Select triplet pair carefully with large mini-batch (>= 1800).
2. Pretrain an embedding model as a classifier with softmax-cross-entropy loss.
3. Try to train with other metric losses.
```
This project uses second option.  
Pretrain first and fine-tune the pretrained model with metric losses.  

## How to Make Your Dataset
You have image_list.json file with the format (json) as below.  
```
{
  "Asian/m.0hn95h9/000012_00@en.jpg": {
    "label": 0,
    "x1": 9,
    "y1": 13,
    "x2": 75,
    "y2": 100
  },
  "Asian/m.0hn95h9/000046_00@ja.jpg": {
    "label": 0,
    "x1": 7,
    "y1": 7,
    "x2": 43,
    "y2": 65
  },
  ...
}
```
The `key` is a relative path of a face image.   
The `value` of the key contains label number and bounding box that indicates exact face location.  
The bounding box [x1, y1, x2, y2] is [left, top, right, bottom] respectively.  
We generated the bounding box using [[11]](https://github.com/blaueck/tf-mtcnn).  

## Common Settings
Execute the command `export PYTHONPATH=$(pwd)` first.  
Set 'img_root_path' option to know where the images are located.  
Set 'train_file' option saved with the format as mentioned above.  
Set 'num_identity' option that is the number of face identities in the 'train_file'.  

## Recommendation Steps for Training.
1. Set 'train_classifier' to `True` and 'arc_margin_penalty' to `False`, then run `python train/main.py`.  
2. Set 'arc_margin_penalty' to `True`, then run `python train/main.py`.  
3. Set 'train_classifier' to `False`, then run `python train/main.py`.  

## Why Do I Have To Train With 3 Steps?
As mentioned above, because it is hard to converge using a triplet loss from scratch.  
A triplet training tends to collapse to f(x)=0, when you should not select hard-sample carefully.  
So, train L2-constrained softmax classifier with small face identities first from scratch.  
Then, finetune the trained L2-constrained model using arc margin penalty loss with large face identities.  
Lastly, finetune the trained arc margin penalty model using a triplet loss.  
Actually you don't have to do last step.  
You can use the arc margin penalty model to make face recognition application.  
But if you don't have GPUs with large memory, you only train with small face identities becuase of memory limitation.  
A triplet training does not depends on the number of face identities.  
It only compares embedding distances between examples, so you can save gpu memory and allocate more batch-size.  
Also the first step is needed because of convergence issues on arc margin panlty.  
It is alleviated by pretraining an initial model with softmax.  
On top of that, training the classifier with small(<=2000) face identities is sufficient to get good initialization for arc margin model.  

## TODO LIST
Do ablation strudy for stable learning on large face identity.  

- [ ] *Known as `Center Loss`*, A Discriminative Feature Learning Approach for Deep Face Recognition, Y. Wen et al., ECCV 2016
- [x] *Known as `L2 Softmax`*, L2-constrained Softmax Loss for Discriminative Face Verification, R. Ranjan et al., arXiv preprint arXiv:1703.09507 2017
- [ ] *Known as `Proxy-NCA`*, No Fuss Distance Metric Learning using Proxies, Y. Movshovitz-Attias et al., ICCV 2017
- [ ] Correcting the Triplet Selection Bias for Triplet Loss, B. Yu et al., ECCV 2018
- [x] Global Norm-Aware Pooling for Pose-Robust Face Recognition at Low False Positive Rate, S. Chen et al., arXiv preprint arXiv:1808.00435 2018
- [ ] The Devil of Face Recognition is in the Noise, F. Wang et al., ECCV 2018
- [ ] Co-Mining: Deep Face Recognition with Noisy Labels, X. Wang et al., ICCV 2019
- [ ] SoftTriple Loss: Deep Metric Learning Without Triplet Sampling, Q. Qian et al., ICCV 2019
- [ ] *Known as `Proxy-Anchor`*, Proxy Anchor Loss for Deep Metric Learning, S. Kim et al., CVPR 2020

## References
1. [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)
2. [Deep Face Recognition, VGGFACE](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
3. [RFW Face Dataset](http://www.whdeng.cn/RFW/index.html)
4. [Trillion Pairs](http://trillionpairs.deepglint.com/overview)
5. https://github.com/davidsandberg/facenet/
6. https://github.com/omoindrot/tensorflow-triplet-loss
7. https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset
8. https://github.com/peteryuX/arcface-tf2
9. [Sharing problems I encountered training Arcface models](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/109987)
10. [Help needed: ArcFace in Keras](https://www.reddit.com/r/deeplearning/comments/cg1kev/help_needed_arcface_in_keras)
11. https://github.com/blaueck/tf-mtcnn
12. [Global Norm-Aware Pooling for Pose-Robust Face Recognition at Low False Positive Rate](https://arxiv.org/ftp/arxiv/papers/1808/1808.00435.pdf)