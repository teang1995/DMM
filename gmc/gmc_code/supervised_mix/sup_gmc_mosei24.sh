# echo "** Train representation model"
# CUDA_VISIBLE_DEVICES=6 python main_supervised.py -f with experiment.scenario="mosei" experiment.stage="train_model" 

# echo "** Evaluate GMC - Classification"
# CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.evaluation_mods=[0] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_24"
# CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_24"
# CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.evaluation_mods=[2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_24"
# CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_24"

echo "** Evaluate GMC - Classification"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_24"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_24"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_24"