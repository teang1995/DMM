# must be tuned : mix_schedule, single_mix, all_mix, multi_mix, beta_params,  learning_rate, temperature,
for seed in 0
do
for uni in 0.75
do
for all in 0.5 0.75 1.0
do
for mm in 0.3
do
for mix_sche in 0.0
do

CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" model.infer_mixed=False model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128

CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128

done
done
done
done
done

for seed in 0
do
for uni in 0.5 1.0
do
for all in 0.75
do
for mm in 0.3
do
for mix_sche in 0.0
do

CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" model.infer_mixed=False model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128

CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128
CUDA_VISIBLE_DEVICES=3 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model.model="mmix_${uni}_${all}_${mm}_${mix_sche}" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=$mix_sche model.single_mix=$uni model.all_mix=$all model.multi_mix=$mm model_train.batch_size=128

done
done
done
done
done