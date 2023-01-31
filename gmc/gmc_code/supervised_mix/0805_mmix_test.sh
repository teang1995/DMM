# must be tuned : mix_schedule, single_mix, all_mix, multi_mix, beta_params,  learning_rate, temperature,
for seed in 0
do
for sm in 1.0 0.0
do

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" model.infer_mixed=False model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.model="mmix_${sm}_${am}" model.model="mmix_${sm}_${am}" model_train.epochs=1 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=0.0 model.single_mix=$sm model.all_mix=$sm model.multi_mix=$sm model_train.batch_size=128

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[1.0,1.0,1.0] model.beta_params=[0.5,0.5] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_${sm}_${am}" model.model="mmix_${sm}_${am}" model_train.epochs=1 experiment.seed=$seed model.mixup_type="replace" \
        model.mix_schedule=0.0 model.single_mix=$sm model.all_mix=$sm model.multi_mix=$sm model_train.batch_size=128

done
done