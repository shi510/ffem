import tensorflow as tf
import numpy as np
import example.train.utils as utils
import imgaug as ia
import imgaug.augmenters as iaa

def make_RFW_tfdataset(root_path, race_list, num_id, batch_size, img_shape, onehot=False):
    img_pathes, img_labels = utils.read_RFW_train_list(root_path, race_list, num_id)
    ds = tf.data.Dataset.from_tensor_slices((img_pathes, img_labels))
    ds = ds.shuffle(len(img_pathes))
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
            #         # search either for all edges or for directed edges,
            #         # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ])
                ],
                random_order=True
            )
        ]
        # random_order=True
    )
    def _preprocess_image(image):
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_shape)
        image = tf.cast(image, tf.uint8)
        [image, ] = tf.numpy_function(lambda img : seq(image=img), [image], [tf.uint8])
        image = tf.cast(image, tf.float32)
        image -= 127.5
        image /= 128 # normalize to [-1,1] range
        return image

    def _load_and_preprocess_image(img_path, label):
        img_path = root_path+'/'+img_path
        image = tf.io.read_file(img_path)
        return _preprocess_image(image), label
    ds = ds.map(_load_and_preprocess_image)
    num_class = None
    if onehot:
        num_class = len(race_list) * num_id
        ds = ds.map(lambda img, label : (img, tf.one_hot(label, num_class)))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds, num_class

