# M2 mix tune
# potential candidate: mix_schedule

for seed in 0
do
for uni in 0.0
do
for all in 0.0
do
for mm in 0.15
do
for temperature in 0.2 0.25 0.3
do
for m2mix_type in both
do
for beta in 2.0 1.5 1.0
do

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[-1] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

done
done
done
done
done
done
done


for seed in 0
do
for uni in 0.0
do
for all in 0.0
do
for mm in 0.2
do
for temperature in 0.15
do
for m2mix_type in both
do
for beta in 1.5 1.75 1.25
do

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[-1] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

done
done
done
done
done
done
done

for seed in 0
do
for uni in 0.0
do
for all in 0.0
do
for mm in 0.2
do
for temperature in 0.2 0.25 0.3
do
for m2mix_type in both
do
for beta in 1.75 1.25
do

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[-1] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type

done
done
done
done
done
done
done