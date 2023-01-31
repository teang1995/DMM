# must be tuned : mix_schedule, single_mix, all_mix, multi_mix, beta_params,  learning_rate, temperature,
#beta 0.5

for seed in 0
do
for uni in 0.25 0.5
do
for all in 0.25 0.5
do
for mm in 0.3
do
for mix_sche in 0.0
do
for temperature in 0.3
do
for learning_rate in 0.001 0.0007 0.0005
do

CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" model.infer_mixed=False model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[1.0,1.0] \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.learning_rate=$learning_rate

CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[1.0,1.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.learning_rate=$learning_rate
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[1.0,1.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.learning_rate=$learning_rate
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[1.0,1.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.learning_rate=$learning_rate
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[1.0,1.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.learning_rate=$learning_rate
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[1.0,1.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.learning_rate=$learning_rate
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[1.0,1.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.learning_rate=$learning_rate
CUDA_VISIBLE_DEVICES=7 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[1.0,1.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model.model="mmix_${uni}_${all}_${mm}_${temperature}_0.5_${learning_rate}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128 model_train.temperature=$temperature model_train.learning_rate=$learning_rate

done
done
done
done
done
done
done