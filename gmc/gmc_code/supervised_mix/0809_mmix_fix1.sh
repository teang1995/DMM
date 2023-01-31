# must be tuned : mix_schedule, single_mix, all_mix, multi_mix, beta_params,  learning_rate, temperature,
#beta 0.5

for seed in 0
do
for uni in 0.5
do
for all in 0.5
do
for mm in 0.1 0.2
do
for temperature in 0.2 0.3 0.5
do
for m2mix_type in pos
do

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" model.infer_mixed=False model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.4,0.4] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.m2mix_type=$m2mix_type

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.4,0.4] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.4,0.4] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.4,0.4] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.4,0.4] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.4,0.4] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.4,0.4] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.m2mix_type=$m2mix_type
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.4,0.4] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.4" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.m2mix_type=$m2mix_type

done
done
done
done
done
done