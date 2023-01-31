import sacred

########################
# Machine Ingredient  #
########################

machine_ingredient = sacred.Ingredient("machine")


@machine_ingredient.config
def machine_config():
    #m_path = "/disk/changdae/gmc/gmc_code/unsupervised/evaluation/gmc_mhd/log_0"
    m_path = "/disk/changdae/gmc/gmc_code/unsupervised"