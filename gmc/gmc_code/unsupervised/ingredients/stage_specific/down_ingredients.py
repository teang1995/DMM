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
    epochs = 100
    learning_rate = 1e-3
    snapshot = 25
    checkpoint = None