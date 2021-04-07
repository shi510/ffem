config = {
    'mixed_precision': False,
    
    'enable_prune': False,
    'prune_params':{
        'initial_sparsity': 0.3,
        'final_sparsity': 0.8,
        'begin_step': 10000,
        'end_step': 30000
    },
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

    #
    # If batch_size and batch_division are 1024 and 8 respectively, the model is created with 128 batch size (1024 / 8).
    # Then, the gradients are accumulated for 8 times and the accumulated gradients are updated.
    #
    'batch_size' : 1024,
    'batch_division': 8,
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
    # 2. ProxyNCA
    # 3. AdditiveAngularMargin (known as ArcFace)
    #
    'loss': 'CenterSoftmax',
    'loss_param':{
        'CenterSoftmax':{
            'scale': 30,
            'center_lr': 1e-2,
            'center_weight': 1e-3,
        },
        'ProxyNCA':{
            'scale': 30,
            'proxy_lr': 1e-2
        },
        'AdditiveAngularMargin':{
            'scale': 30,
            'margin': 0.5
        }
    },

    'eval':{
        'metric': 'cos',
        'recall': [1]
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
    'train_file': 'your_train.tfrecord',
    'test_files': ['your_test.tfrecord'],

    #
    # Set maximum face ID in 'tfrecord_file'.
    #
    'num_identity': 4000
}
