config = {
    # Save trained model named with 'model_name'.h5.
    'model_name': 'face_model',
    'saved_model': 'face_model.h5',
    # Which do you want to execute with Keras or Eager Mode?
    # Eager mode is for debugging.
    'use_keras': True,
    # You should train your model as classifier first for stable metric learning.
    # It is trained with softmax-cross-entropy-with-logits loss function.
    # Triplet-loss is easy to collapse to f(x)=0, when should not select hard-sample carefully.
    # Turn this option off after training the classifier is done.
    'train_classifier': True,
    'batch_size' : 128,
    'shape' : [160, 160, 3],
    # Choose one of below: 
    # 1. MobileNetV2
    # 2. ResNet18
    'model' : 'MobileNetV2',
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
    'metric_loss' : 'triplet_loss.original_triplet_loss',
    'embedding_dim': 128,
    'optimizer' : 'adam',
    'epoch' : 1000,
    'learning_rate' : 1e-4,
    'learning_rate_decay' : 0.96,
    'dataset': 'RFW',

    # Describe dataset configuration below.
    '__RFW':{
        'train_path': '/your/rfw/datset/BUPT-Balancedface/images/race_per_7000',
        'test_path': '/your/rfw/datset/RFW/images/test/data',
        # Possible race list is :
        # 1. African
        # 2. Asian
        # 3. Caucasian
        # 4. Indian
        'race_list': ['African', 'Asian', 'Caucasian', 'Indian'],

        # Number of faces in each race.
        # Set the value to None if you want to include all faces in each rase set.
        # The number of all identities is 'num_identity' * number of items in 'race_list'.
        'num_identity': None,
    }
}
