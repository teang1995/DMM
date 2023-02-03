GMC_FLAG=$1
for SEED in 0 1 2 3 4
do

CUDA_VISIBLE_DEVICES=0 python main_supervised.py -f with experiment.scenario="mosei" \
experiment.stage="train_model" experiment.model="gmc${gmc}_seed${SEED}" model.model="gmc${gmc}_seed${SEED}" \
experiment.seed=$SEED  model.ldmm=0.0 model.gmc=$GMC_FLAG

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] \
experiment.stage="evaluate_downstream_classifier" experiment.model="gmc${gmc}_seed${SEED}" model.model="gmc${gmc}_seed${SEED}" \
experiment.seed=$SEED model.ldmm=0.0 model.gmc=$GMC_FLAG

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] \
experiment.stage="evaluate_downstream_classifier" experiment.model="gmc${gmc}_seed${SEED}" model.model="gmc${gmc}_seed${SEED}" \
experiment.seed=$SEED model.ldmm=0.0 model.gmc=$GMC_FLAG

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] \
experiment.stage="evaluate_downstream_classifier" experiment.model="gmc${gmc}_seed${SEED}" model.model="gmc${gmc}_seed${SEED}" \
experiment.seed=$SEED model.ldmm=0.0 model.gmc=$GMC_FLAG

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] \
experiment.stage="evaluate_downstream_classifier" experiment.model="gmc${gmc}_seed${SEED}" model.model="gmc${gmc}_seed${SEED}" \
experiment.seed=$SEED model.ldmm=0.0 model.gmc=$GMC_FLAG

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] \
experiment.stage="evaluate_downstream_classifier" experiment.model="gmc${gmc}_seed${SEED}" model.model="gmc${gmc}_seed${SEED}" \
experiment.seed=$SEED model.ldmm=0.0 model.gmc=$GMC_FLAG

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] \
experiment.stage="evaluate_downstream_classifier" experiment.model="gmc${gmc}_seed${SEED}" model.model="gmc${gmc}_seed${SEED}" \
experiment.seed=$SEED model.ldmm=0.0 model.gmc=$GMC_FLAG

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] \
experiment.stage="evaluate_downstream_classifier" experiment.model="gmc${gmc}_seed${SEED}" model.model="gmc${gmc}_seed${SEED}" \
experiment.seed=$SEED model.ldmm=0.0 model.gmc=$GMC_FLAG

done