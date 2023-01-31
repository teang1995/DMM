
for seed in 0
do

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="mmix_nl_01" \
        experiment.seed=$seed experiment.log_name="0802_no_label" experiment.use_label=False \
        model.model="mmix_nl_01" model.dirichlet_params=[0.1,0.1,0.1] model.mixup_type="replace" \
        model_train.epochs=2 down_train.epochs=2

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="mmix_nl_01" \
        experiment.seed=$seed experiment.log_name="0802_no_label" experiment.use_label=False \
        model.model="mmix_nl_01" model.dirichlet_params=[0.1,0.1,0.1] model.mixup_type="replace" \
        model_train.epochs=2 down_train.epochs=2

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_01" \
        experiment.seed=$seed experiment.log_name="0802_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0] \
        model.model="mmix_nl_01" model.dirichlet_params=[0.1,0.1,0.1] model.mixup_type="replace" \
        model_train.epochs=2 down_train.epochs=2

done