# CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
# -f with experiment.scenario="mosei" \
#         experiment.model="gmc32" \
#         model.model='gmc32'
#         experiment.stage="train_model" \
#         model_train.batch_size=32 \
#         down_train.batch_size=32
# echo "** Evaluate GMC - Classification"
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc32" model.model='gmc32'
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc32" model.model='gmc32'
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc32" model.model='gmc32'
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc32" model.model='gmc32'

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc32" model.model='gmc32'
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc32" model.model='gmc32'
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc32" model.model='gmc32'