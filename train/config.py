config = {

    #
    # This is for experiment.
    # It often results in NAN value during training.
    #
    'mixed_precision': False,
    
    #
    # This is for experiment and it is used for fine-tuning.
    #
    'enable_prune': False,
    'prune_params':{
        'initial_sparsity': 0.3,
        'final_sparsity': 0.5,
        'begin_step': 0,
        'end_step': 30000
    },

    #
    # This is for experiment and it is used for fine-tuning.
    #
    'enable_quant_aware' : False,

    #
    # Save trained model named with 'model_name'.h5.
    # The best model at each epoch is saved to the folder ./checkpoint/'model_name'.
    #
    'model_name': 'ResNet50_centerloss_yourdataset_4000',

    #
    # Restore weights of backbone network.
    # It restores only weights of backbone network.
    #
    'saved_backbone': '',

    #
    # The checkpoint option is different from saved_backbone option.
    # It restores the entire weights of a custom model.
    # So it overrides the weights of saved_backbone with the weights in checkpoint if you feed both options.
    # The path should indicate a directory containing checkpoint files.
    #
    'checkpoint': '',

    'batch_size' : 256,

    #
    # It is for training with large batch size on a limited GPU memory.
    # It accumulates gradients for 'num_grad_accum' times, then applies accumulated gradients.
    # The total batch size is 'batch_size' * 'num_grad_accum'.
    # ex) 'num_grad_accum' = 4 and 'batch_size' = 256, then total batch size is 1024.
    #
    'num_grad_accum': 4,
    'shape' : [112, 112, 3],

    #
    # Choose one of below: 
    # 1. MobileNetV2
    # 2. MobileNetV3
    # 3. EfficientNetB3
    # 4. ResNet50
    #
    'model' : 'ResNet50',
    'embedding_dim': 512,

    #
    # 1. SoftmaxCenter (known as Center Loss)
    # 2. AngularMargin (known as ArcFace)
    # 3. GroupAware (known as GroupFace)
    #
    'loss': 'AngularMargin',
    'loss_param':{
        'SoftmaxCenter':{
            'scale': 30,
            'center_loss_weight': 1e-3,
        },
        'AngularMargin':{
            'scale': 60,
            'margin': 0.5
        },
        'GroupAware':{
            'scale': 60,
            'margin': 0.5,
            'num_groups': 4,
            'intermidiate_dim': 256,
            'group_loss_weight': 0.1
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
