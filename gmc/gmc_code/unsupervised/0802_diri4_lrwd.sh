for lr in 1e-3 5e-4 1e-4
do
for wd in 0.0 1e-3 1e-4
do

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[1] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[2] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,1] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[1,2] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,2] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_nl_20" \
        experiment.seed=0 experiment.log_name="0802_no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,1,2] \
        model.model="mmix_nl_20" model.dirichlet_params=[2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        experiment.evaluation_mods=[0] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        experiment.evaluation_mods=[1] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        experiment.evaluation_mods=[2] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        experiment.evaluation_mods=[0,1] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        experiment.evaluation_mods=[1,2] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        experiment.evaluation_mods=[0,2] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="mmix_l_20" \
        experiment.seed=0 experiment.log_name="0802_label" experiment.use_label=True \
        experiment.evaluation_mods=[0,1,2] \
        model.model="mmix_l_20" model.dirichlet_params=[2.0,2.0,2.0,2.0] model.mixup_type="replace" \
        model_train.epochs=100 down_train.epochs=50

done
done