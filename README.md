# FFEM  
FFEM stands for Face Feature Embedding Module.  
The Tensorflow version should be v2.4.0 or later for mixed precision training.  
The tensorflow-addons is needed for tfa.images.  

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

## Transform the json file into TFRECORD
Input pipeline bottleneck increases training time.  
Reading data from a large file sequentially is better than reading a lot of small sized data randomly.  
Try the command below, it generates [name].tfrecord file from the above json file.  
```
python generate_tfrecord/main.py --root_path [path] --json_file [path] --output [name]
```

## Common Settings
Execute the command `export PYTHONPATH=$(pwd)` first.  
Set 'img_root_path' arg to know where the images are located.  
Set 'train_file' arg saved with the format as mentioned above.  
Set 'num_identity' arg that is the number of face identities in the 'train_file'.  

## Recommendation Steps for Training.
1. Set 'train_classifier' to `True` and 'arc_margin_penalty' to `False`, then run `python train/main.py`.  
2. Set 'arc_margin_penalty' to `True`, then run `python train/main.py`.  

## Why Do I Have To Train With 2 Steps?
If your face dataset quality is not good, your trained model also is not good.  
It maybe good quality if your dataset has 200+ average images per identity.  
In the case which you want to train asian dataset, trillion pairs, it has 30+ average images per identity.  
So the quality of the dataset is not good, we should fine-tune a model from pretrained face embedding model.  
There are good face dataset, for example, vggface2 dataset that has 300+ average images per identity.  
Consequently you have to train with 2 steps.  
The training steps is described below.  
On top of that, small(<=2000) face identities are sufficient to get good initialization when you are training the first step.  

## Training Conditions
```
1. Train MobiletNetV3 architecture from scratch as details below:
 - L2-constrained softmax with 30 scale factor
 - ReLU6 activation (Default in this repository)
 - ADAM optimizer with 1e-4 learning rate
 - 50 epochs
 - VGGFACE2 dataset with 2K identities

2. Train the above pretrained model as details below:
 - additive angular margin loss, 0.5 margin and 60 scale factor
 - SGD momentum nesterov optimizer with 1e-3 learning rate
 - 10~15 epochs
 - trillion pairs dataset with large identities
```

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
- [ ] Relational Deep Feature Learning for Heterogeneous Face Recognition, M. Cho et al., IEEE 2020

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