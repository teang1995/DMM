for bs in 128 24
do

CUDA_VISIBLE_DEVICES=7 python main_supervised.py -f with experiment.scenario="mosei" experiment.stage="train_model" experiment.model="gmc_${bs}" experiment.seed=0 model_train.batch_size=$bs

CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_${bs}" experiment.seed=0 model_train.batch_size=$bs
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_${bs}" experiment.seed=0 model_train.batch_size=$bs
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_${bs}" experiment.seed=0 model_train.batch_size=$bs
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_${bs}" experiment.seed=0 model_train.batch_size=$bs
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_${bs}" experiment.seed=0 model_train.batch_size=$bs
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_${bs}" experiment.seed=0 model_train.batch_size=$bs
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_${bs}" experiment.seed=0 model_train.batch_size=$bs

done