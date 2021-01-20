config = {
    'mixed_precision': True,
    #
    # Save trained model named with 'model_name'.h5.
    # The best model at each epoch is saved to the folder ./checkpoint/'model_name'.
    #
    'model_name': 'face_angular_softmax_7000',

    #
    # Restore trained weights.
    # The architecture must be same with 'model' option.
    # checkpoint folder or keras saved file including extension.
    #
    'saved_model': '',

    #
    # Which do you want to execute with Keras or Eager Mode?
    # Eager mode is for debugging.
    #
    'use_keras': True,

    'arc_margin_penalty': False,
    'batch_size' : 512,
    'shape' : [112, 112, 3],

    #
    # If 'saved_model' not exsits, then it will be built with this architecture.
    # Choose one of below: 
    # 1. MobileNetV2
    # 2. MobileNetV3
    # 3. EfficientNetB3
    # 4. ResNet50
    #
    'model' : 'ResNet50',
    'embedding_dim': 512,

    #
    # There are two options.
    #  1. Adam
    #  2. AdamW
    #  3. SGD with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'AdamW',

    #
    # initial learning rate.
    #
    'lr' : 1e-4,

    #
    # lr * decay_rate ^ (steps / decay_steps)
    #
    'lr_decay': False,
    'lr_decay_steps' : 25000,
    'lr_decay_rate' : 0.96,


    #
    # training dataset generated from generate_tfrecord/main.py
    # See README.md
    #
    'tfrecord_file': 'your.tfrecord',
 
    #
    # Set maximum face ID in 'tfrecord_file'.
    #
    'num_identity': 2000
}
