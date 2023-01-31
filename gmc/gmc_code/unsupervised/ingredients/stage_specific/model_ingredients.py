import sacred
#! Edit this file for adding and managing custom argument
#! model.dirichlet=[2.0,2.0,2.0,2.0]

###########################
#        Model            #
###########################
model_ingredient = sacred.Ingredient("model")


@model_ingredient.config
def gmc_mhd():
    model = "gmc"
    common_dim = 64
    latent_dim = 64
    loss_type = "infonce"  # "joints_as_negatives"
    mixup_type = 'no'  #! choices: ["no", "replace"]
    mixup_dist = 'dirichlet' #! choices: ["dirichlet", "beta"]
    dirichlet_params = [2.0,2.0,2.0,2.0]
    beta_params = [2.0,2.0]

##############################
#       Model  Train         #
##############################
model_train_ingredient = sacred.Ingredient("model_train")

#! original params
#! BS = 64
#! epochs = 100
@model_train_ingredient.config
def gmc_mhd_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 512
    num_workers = 8

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-3
    weight_decay = 0.0 #! simclr default 1e-4
    snapshot = 50
    checkpoint = None
    temperature = 0.1