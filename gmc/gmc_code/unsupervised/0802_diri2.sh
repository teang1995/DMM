
for seed in 0 1 2 3 4
do

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0] \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[1] \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[2] \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,1] \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[1,2] \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,2] \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_05" \
        experiment.seed=$seed experiment.log_name="0803_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,1,2] \
        model.model="mmix_nl_05" model.dirichlet_params=[0.5,0.5,0.5] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50
done