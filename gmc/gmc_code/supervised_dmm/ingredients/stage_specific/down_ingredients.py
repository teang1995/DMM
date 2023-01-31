import sacred

###################################
#       Downstream  Train         #
###################################

down_train_ingredient = sacred.Ingredient("down_train")


####################
#  Classification  #
####################

@down_train_ingredient.config
def mosi():
    # Dataset parameters
    batch_size = 128
    num_workers = 2

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 25
    checkpoint = None


@down_train_ingredient.config
def mosei():
    # Dataset parameters
    batch_size = 128
    num_workers = 2

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 25
    checkpoint = None