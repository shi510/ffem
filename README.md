# FFEM  
FFEM stands for Face Feature Embedding Module.  
This project includes following implementations:  
1. ArcFace  
2. GroupFace  
3. CenterLoss  

## Requirements
```
tensorflow==2.4.1
tensorflow-addons==0.12.1
tensorflow-model-optimization==0.5.0
numpy==1.19.5
```

## How to Make Your Dataset
You have image_list.json file with the format (json) as below.  
```
{
  "Caucacian/a12/frontal1.jpg": {
    "label": 0,
    "x1": 9,
    "y1": 13,
    "x2": 75,
    "y2": 100
  },
  "Asian/p14/profile2.jpg": {
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
python generate_tfrecord/main.py --root_path [path] --json_file [path] --output [name].tfrecord
```

## Common Settings
Execute the command `export PYTHONPATH=$(pwd)` on linux and `$env:$PYTHONPATH=$pwd` on windows 10 powershell.  

## Recommendation Steps for Training.
1. Train a model with 'loss'=`SoftmaxCenter` on VGGFACE2 dataset.  
2. Train the pretrained model from first step with 'loss'=`AngularMargin` on large identity dataset.  
The training command is `python train/main.py`.  


## Results
|                       |        ResNet50        |        ResNet50           |
|-----------------------|------------------------|---------------------------|
| Recall @ 1, African   | 51%                    | 55%                       |
| Recall @ 1, Asian     | 83%                    | 84%                       |
| Recall @ 1, Caucacian | 69%                    | 74%                       |
| Recall @ 1, Indian    | 69%                    | 72%                       |
| Recall @ 1, VGGFace2  | 89%                    | 95%                       |
| Epoch                 | 50                     | 70                        |
| Batch Size            | 2048                   | 2048                      |
| Embedding Size        | 512                    | 512                       |
| Feature Pooling       | *GNAP                  | *GNAP                     |
| Loss Type             | AngularMargin(arcface) | AngularMargin(arcface)    |
| Scale                 | 60                     | 60                        |
| LR                    | SGD@1e-1               | SGD@1e-1                  |
| # of Identity         | 93979                  | 100979                    |
| Dataset               | Trillion Pairs         | Trillion Pairs + VGGFACE2 |

**RFW and VGGFACE2 are used for testing*  
**All models are pretrained on VGGFACE2 train-set*  
**Global Norm-Aware Pooling (GNAP) is used for pooling last spatial features of convolution layer.*  

## TODO LIST

- [x] *Known as `Center Loss`*, A Discriminative Feature Learning Approach for Deep Face Recognition, Y. Wen et al., ECCV 2016
- [x] *Known as `L2 Softmax`*, L2-constrained Softmax Loss for Discriminative Face Verification, R. Ranjan et al., arXiv preprint arXiv:1703.09507 2017
- [x] Global Norm-Aware Pooling for Pose-Robust Face Recognition at Low False Positive Rate, S. Chen et al., arXiv preprint arXiv:1808.00435 2018
- [ ] The Devil of Face Recognition is in the Noise, F. Wang et al., ECCV 2018
- [ ] Co-Mining: Deep Face Recognition with Noisy Labels, X. Wang et al., ICCV 2019
- [x] ArcFace: Additive Angular Margin Loss for Deep Face Recognition, J. Deng et al., CVPR 2019
- [ ] Relational Deep Feature Learning for Heterogeneous Face Recognition, M. Cho et al., IEEE 2020
- [ ] Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces, J. Deng et al., ECCV 2020
- [x] GroupFace: Learning Latent Groups and Constructing Group-based Representations for Face Recognition, Y. Kim et al., CVPR 2020

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
13. [GroupFace: Learning Latent Groups and Constructing Group-based Representations for Face Recognition](https://arxiv.org/pdf/2005.10497.pdf)