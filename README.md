# FFEM  
FFEM stands for Face Feature Embedding Module.  
This project is tested on tensorflow-v2.3.0.  

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
We generated the bounding box using [[10]](https://github.com/blaueck/tf-mtcnn).  

## Common Settings
Execute the command `export PYTHONPATH=$(pwd)` first.  
Set 'img_root_path' option to know where the images are located.  
Set 'train_file' option saved with the format as mentioned above.  
Set 'num_identity' option that is the number of face identities in the 'train_file'.  

## First Step (Classifier)
1. Choose 'batch_size' for consideration of memory capacity.  
2. Set 'train_classifier' to `True`.  
3. Run `python example/train/main.py`.  

## Last Step (Metric Learning)
1. Set 'train_classifier' to `False`.  
2. Choose 'metric_loss' option.  
3. Run `python example/train/main.py`.  

## References
1. [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)
2. [Deep Face Recognition, VGGFACE](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
3. [RFW Face Dataset](http://www.whdeng.cn/RFW/index.html)
4. https://github.com/davidsandberg/facenet/
5. https://github.com/omoindrot/tensorflow-triplet-loss
6. https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset
7. https://github.com/peteryuX/arcface-tf2
8. [Sharing problems I encountered training Arcface models](https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/109987)
9. [Help needed: ArcFace in Keras](https://www.reddit.com/r/deeplearning/comments/cg1kev/help_needed_arcface_in_keras)
10. https://github.com/blaueck/tf-mtcnn
