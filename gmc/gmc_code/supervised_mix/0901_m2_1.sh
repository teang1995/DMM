# must be tuned : mix_schedule, single_mix, all_mix, multi_mix, beta_param,  learning_rate, temperature,
#beta 0.5

for seed in 0
do
for uni in 0.0
do
for all in 0.0
do
for mm in 0.3 0.4 0.5
do
for temperature in 0.1 0.2 0.3
do
for m2mix_type in pos both
do
for beta in 1.0
do

CUDA_VISIBLE_DEVICES=1 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,1] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,2] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[1,2] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,1,2] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[1] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[2] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

done
done
done
done
done
done
done