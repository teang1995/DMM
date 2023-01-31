# M2 mix tune
# potential candidate: mix_schedule

for seed in 1 2 3 4
do
for uni in 0.0
do
for all in 0.0
do
for mm in 0.2
do
for temperature in 0.25
do
for m2mix_type in both
do
for beta in 2.0
do

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0] experiment.wb_proj='1009_MMIX_qualitative' \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type experiment.vis='non'

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[1] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type experiment.vis='non'

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[2] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type experiment.vis='non'

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0,1] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type experiment.vis='non'

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0,2] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type experiment.vis='non'

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[1,2] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_${beta}_${m2mix_type}_${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.beta_param=$beta model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model.m2mix_type=$m2mix_type experiment.vis='non'

done
done
done
done
done
done
done