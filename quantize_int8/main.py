import argparse
import os

import tensorflow as tf

def convert_tflite_int8(model, ds, output_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_dataset_gen():
        for n, (x, _ )in enumerate(ds.take(5000)):
            if n % 100 == 0:
                print(n)
            # Get sample input data as a numpy array in a method of your choosing.
            # The batch size should be 1.
            yield [x]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8  # or tf.int8
    tflite_quant_model = converter.convert()
    with open(output_name, 'wb') as f:
        f.write(tflite_quant_model)


def input_pipeline(dataset_file, input_shape):
    def _read_tfrecord(serialized):
        description = {
            'jpeg': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64),
            'x1': tf.io.FixedLenFeature((), tf.int64),
            'y1': tf.io.FixedLenFeature((), tf.int64),
            'x2': tf.io.FixedLenFeature((), tf.int64),
            'y2': tf.io.FixedLenFeature((), tf.int64)
        }
        example = tf.io.parse_single_example(serialized, description)
        image = tf.io.decode_jpeg(example['jpeg'], channels=3)
        label = example['label']
        box = [
            tf.cast(example['y1'], tf.float32),
            tf.cast(example['x1'], tf.float32),
            tf.cast(example['y2'], tf.float32),
            tf.cast(example['x2'], tf.float32)]
        return image, label, box

    def _load_and_preprocess_image(image, label, box):
        # shape = [Height, Width, Channel]
        shape = tf.shape(image)
        # shape = [Height, Height, Width, Width]
        shape = tf.repeat(shape, [2, 2, 0])
        # shape = [Height, Width, Height, Width]
        shape = tf.scatter_nd([[0], [2], [1], [3]], shape, tf.constant([4]))
        # Normalize [y1, x1, y2, x2] box by width and height.
        box /= tf.cast(shape, tf.float32)
        image = tf.cast(image, tf.float32)
        return image, label, box


    def _normalize(x: tf.Tensor):
        # Normalize images to the range [0, 1].
        return x / 255.

    test_ds = tf.data.TFRecordDataset(dataset_file)
    test_ds = test_ds.map(_read_tfrecord)
    test_ds = test_ds.shuffle(10000)
    test_ds = test_ds.map(_load_and_preprocess_image)
    test_ds = test_ds.map(
        lambda x, label, box: (tf.image.crop_and_resize([x], [box], [0], input_shape)[0], label))
    test_ds = test_ds.batch(1)
    test_ds = test_ds.map(lambda img, label : (_normalize(img), label))

    return test_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras_model', type=str, required=True,
        help='trained keras model path')
    parser.add_argument('--dataset', type=str, required=True,
        help='calibration dataset (tfrecord)')
    parser.add_argument('--image_size', type=str, required=True,
        help='image width and height. ex) 112,112')
    args = parser.parse_args()
    img_size = args.image_size.split(',')
    width = int(img_size[0])
    height = int(img_size[1])
    dataset = input_pipeline(args.dataset, (width, height))
    net = tf.keras.models.load_model(args.keras_model)
    convert_tflite_int8(net, dataset, os.path.splitext(args.keras_model)[0] + '.tflite')
