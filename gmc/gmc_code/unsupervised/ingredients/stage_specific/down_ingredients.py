import sacred

###################################
#       Downstream  Train         #
###################################
# only for clsasfier
down_train_ingredient = sacred.Ingredient("down_train")

#! original params
#! BS = 64
#! epochs = 100
@down_train_ingredient.config
def mhd():
    # Dataset parameters
    batch_size = 512
    num_workers = 8

    # Training Hyperparameters (디폴트는 100인데, 논문에 50에폭이라 나와있음.)
    epochs = 100
    learning_rate = 1e-3
    snapshot = 25
    checkpoint = None