config = {
    'mixed_precision': False,
    #
    # Save trained model named with 'model_name'.h5.
    # The best model at each epoch is saved to the folder ./checkpoint/'model_name'.
    #
    'model_name': 'ResNet50_centerloss_yourdataset_4000',

    #
    # Restore trained weights.
    # The architecture must be same with 'model' option.
    # checkpoint folder or keras saved file including extension.
    #
    'saved_model': '',

    'batch_size' : 256,
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
    # 1. CenterSoftmax (known as Center Loss)
    # 2. ProxySoftmax (Softmax with ProxyNCA)
    # 3. AddictiveMargin (known as ArcFace)
    #
    'loss': 'CenterSoftmax',
    'loss_param':{
        'CenterSoftmax':{
            'scale': 30,
            'center_lr': 1e-2,
            'center_weight': 1e-3,
        },
        'ProxySoftmax':{
            'scale': 30,
            'proxy_lr': 1e-2,
            'proxy_weight': 1e-3,
        },
        'AddictiveMargin':{
            'scale': 30,
            'margin': 0.5
        }
    },

    #
    # There are two options.
    #  1. Adam
    #  2. AdamW
    #  3. SGD with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'AdamW',
    'epoch' : 40,

    #
    # initial learning rate.
    #
    'lr' : 1e-4,

    #
    # lr * decay_rate ^ (steps / decay_steps)
    #
    'lr_decay': False,
    'lr_decay_steps' : 10000,
    'lr_decay_rate' : 0.9,


    #
    # training dataset generated from generate_tfrecord/main.py
    # See README.md
    #
    'test_files': ['your_test.tfrecord'],

    #
    # Set maximum face ID in 'tfrecord_file'.
    #
    'num_identity': 4000
}
