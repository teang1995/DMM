
for seed in 0 1 2 3 4
do

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True model.model="gmc_l" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True model.model="gmc_l" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[0] \
        model.model="gmc_l" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[1] \
        model.model="gmc_l" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[2] \
        model.model="gmc_l" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[0,1] \
        model.model="gmc_l" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[1,2] \
        model.model="gmc_l" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[0,2] \
        model.model="gmc_l" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l" \
        experiment.seed=$seed experiment.log_name="label" experiment.use_label=True \
        experiment.evaluation_mods=[0,1,2] \
        model.model="gmc_l" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False model.model="gmc_nl" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False model.model="gmc_nl" \
        model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0] \
        model.model="gmc_nl" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[1] \
        model.model="gmc_nl" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[2] \
        model.model="gmc_nl" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,1] \
        model.model="gmc_nl" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[1,2] \
        model.model="gmc_nl" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,2] \
        model.model="gmc_nl" model_train.epochs=100 down_train.epochs=100

CUDA_VISIBLE_DEVICES=6 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl" \
        experiment.seed=$seed experiment.log_name="no_label" experiment.use_label=False \
        experiment.evaluation_mods=[0,1,2] \
        model.model="gmc_nl" model_train.epochs=100 down_train.epochs=100

done