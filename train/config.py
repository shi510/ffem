config = {

    #
    # Save trained model named with 'model_name'.h5.
    # The best model named checkpoint + 'model_name' is saved at each epoch.
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

    #
    # Recommendation settings.
    #  1. Set 'train_classifier' to True.
    #  2. Set 'arc_margin_penalty' to False and train.
    #  3. Set 'arc_margin_penalty' to True and train.
    #  4. Set 'train_classifier' to False and train.
    #
    # Why do i have to train with 3 steps?
    # Because it is hard to converge using a triplet loss from scratch.
    # A triplet training tends to collapse to f(x)=0, when you should not select hard-sample carefully.
    # So, train first with softmax classifier from scratch.
    # Then, train again with arc margin penalty.
    # Lastly, train arc margin penalty model with a triplet loss.
    #
    # Actually You don't have to do last step.
    # You can use the arc margin penalty model to build face embedding application.
    # But if you don't have gpus with large memory, you only train with small face identities becuase of memory limitation.
    # A triplet training does not depends on the number of face identities.
    # It only compares embedding distances between examples.
    # Also the second step is needed because of convergence issues.
    # It is alleviated by pretraining with softmax.
    #
    'train_classifier': True,
    'arc_margin_penalty': False,
    'batch_size' : 192,
    'shape' : [112, 112, 3],

    #
    # If 'saved_model' not exsits, then it will be built with this architecture.
    # Choose one of below: 
    # 1. MobileNetV2
    # 2. MobileNetV3
    # 3. EfficientNetB3
    #
    'model' : 'MobileNetV3',

    #
    # The 'metric_loss' option is enabled when the 'train_classifier' option is turned off.
    # There are two functions for this option.
    #  1. original_triplet_loss
    #  2. adversarial_triplet_loss
    # The original_triplet_loss is that it selects hard samples within mini-batch.
    # The adversarial_triplet_loss is that it is same with original_triplet_loss except for calculating loss.
    # As the name says, two loss functions are combined.
    # One of them try to maximize an anchor-negative distance.
    # The other try to minimize an anchor-positive distance.
    # The adversarial_triplet_loss does not results in zero loss.
    # But original_triplet_loss results in zero loss when it is satisfied by
    #    ||anchor-positive|| < ||anchor-nagative|| + margin.
    #
    'metric_loss' : 'triplet_loss.original_triplet_loss',
    'embedding_dim': 512,

    #
    # There are two options.
    #  1. adam
    #  2. sgd with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'sgd',
    'epoch' : 10000,

    #
    # initial learning rate.
    #
    'lr' : 1e-1,

    #
    # lr * decay_rate ^ (steps / decay_steps)
    #
    'lr_decay': False,
    'lr_decay_steps' : 25000,
    'lr_decay_rate' : 0.96,

    #
    # It should be absolute path that indicates face image file location in 'train_file' contents.
    #
    'img_root_path': '/your/face/image/root_path',

    #
    # See README.md file how to save this file.
    #
    'train_file': './train_list.json',
 
    #
    # Set maximum face ID in 'train_file'.
    # If None, it is set by maximum label from the 'train_file'.
    #
    'num_identity': None
}
