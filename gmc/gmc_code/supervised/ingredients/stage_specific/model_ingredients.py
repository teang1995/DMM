import sacred


###########################
#        Model            #
###########################

model_ingredient = sacred.Ingredient("model")

# MOSI
@model_ingredient.config
def gmc_mosi():
    model = "gmc"
    common_dim = 60
    latent_dim = 60
    loss_type = "infonce"
    mixup_type = 'no'  #! choices: ["no", "replace", "add", "add"]
    mixup_dist = 'dirichlet' #! choices: ["dirichlet", "beta"]
    dirichlet_params = [1.0,1.0,1.0]
    beta_params = [2.0,2.0]
    infer_mixed = False


# MOSEI
@model_ingredient.config
def gmc_mosei():
    model = "gmc"
    common_dim = 60
    latent_dim = 60
    loss_type = "infonce"
    mixup_type = 'no'  #! choices: ["no", "replace", "add", "add"]
    mixup_dist = 'dirichlet' #! choices: ["dirichlet", "beta"]
    dirichlet_params = [1.0,1.0,1.0]
    beta_params = [2.0,2.0]
    infer_mixed = False




##############################
#       Model  Train         #
##############################


model_train_ingredient = sacred.Ingredient("model_train")

# MOSI
@model_train_ingredient.config
def gmc_mosi_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 24
    num_workers = 1

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None
    temperature = 0.3


# MOSEI
@model_train_ingredient.config
def gmc_mosei_train():
    # Dataset parameters
    data_dir = "./dataset/"
    batch_size = 24
    num_workers = 1

    # Training Hyperparameters
    epochs = 40
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None
    temperature = 0.3
