# must be tuned : mix_schedule, single_mix, all_mix, multi_mix, beta_params,  learning_rate, temperature,
#beta 0.5

for seed in 0
do
for uni in 0.5
do
for all in 0.5
do
for mm in 1.0
do
for temperature in 1.0
do
for m2mix_type in pos_jm
do

CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model_train.epochs=1 experiment.seed=$seed \
        model.beta_params=1.0 model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,1] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model_train.epochs=1 experiment.seed=$seed \
        model.beta_params=1.0 model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,2] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model_train.epochs=1 experiment.seed=$seed \
        model.beta_params=1.0 model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[1,2] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model_train.epochs=1 experiment.seed=$seed \
        model.beta_params=1.0 model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,1,2] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model_train.epochs=1 experiment.seed=$seed \
        model.beta_params=1.0 model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model_train.epochs=1 experiment.seed=$seed \
        model.beta_params=1.0 model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[1] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model_train.epochs=1 experiment.seed=$seed \
        model.beta_params=1.0 model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[2] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_test" model_train.epochs=1 experiment.seed=$seed \
        model.beta_params=1.0 model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

done
done
done
done
done
done