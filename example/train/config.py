config = {

    #
    # Save trained model named with 'model_name'.h5.
    #
    'model_name': 'face_angular_softmax_7000',

    #
    # Restore trained weights.
    # The architecture must be same with 'model' option.
    # checkpoint folder or keras saved file including extension.
    #
    'saved_model': './checkpoint',

    #
    # Which do you want to execute with Keras or Eager Mode?
    # Eager mode is for debugging.
    #
    'use_keras': True,

    #
    # You should train your model as classifier first for stable metric learning.
    # It is trained with softmax-cross-entropy-with-logits loss function.
    # Triplet-loss is easy to collapse to f(x)=0, when should not select hard-sample carefully.
    # Turn this option off after training the classifier is done.
    #
    'train_classifier': True,
    'batch_size' : 192,
    'shape' : [128, 128, 3],

    #
    # Choose one of below: 
    # 1. MobileNetV2
    # 2. ResNet18
    #
    'model' : 'MobileNetV2',

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
    'epoch' : 2000,

    #
    # initial learning rate.
    #
    'lr' : 1e-1,

    #
    # lr * decay_rate ^ (steps / decay_steps)
    #
    'lr_decay_steps' : 25000,
    'lr_decay_rate' : 0.96,

    #
    # It should be absolute path that indicates face image file location in 'train_file' contents.
    #
    'img_root_path': '/your/rfw/datset/BUPT-Balancedface/images/race_per_7000',

    #
    # See README.md file how to save this file.
    #
    'train_file': './train_list.json',
 
    #
    # Set maximum face ID in 'train_file'.
    #
    'num_identity': 7000
}
