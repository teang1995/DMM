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
    
    m2mix_type = 'pos_jm'
    mix_stopgrad = 0
    beta_param = 1.0

    single_mix=0.0
    all_mix=0.0
    multi_mix=0.0
    mix_schedule=0.0

# MOSEI
@model_ingredient.config
def gmc_mosei():
    model = "gmc"
    common_dim = 60
    latent_dim = 60
    loss_type = "infonce"
    
    m2mix_type = 'pos_jm'
    mix_stopgrad = 0
    beta_param = 1.0

    single_mix=0.0
    all_mix=0.0
    multi_mix=0.0
    mix_schedule=0.0




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
