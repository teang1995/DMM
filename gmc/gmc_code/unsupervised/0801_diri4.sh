
for seed in 0 1 2 3 4
do

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[1] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[2] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,1] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[1,2] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,2] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,1,2] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[0] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[1] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[2] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[0,1] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[1,2] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[0,2] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=5 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[0,1,2] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=100

done