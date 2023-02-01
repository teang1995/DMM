# for seed in 0 1 2 3 4
# do
CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="train_model" \
        experiment.model="mmix_test" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=True \
        model.model="mmix_test" \
        model.mixup_type="replace" \
        model_train.epochs=1 \
        down_train.epochs=1

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" \
        experiment.model="mmix_test" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=True \
        model.model="mmix_test" \
        model.mixup_type="replace" \
        model_train.epochs=1 \
        down_train.epochs=1

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" \
        experiment.model="mmix_test" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=True \
        experiment.evaluation_mods=[0,1,2,3] \
        model.model="mmix_test" \
        model.mixup_type="replace" \
        model_train.epochs=1 \
        down_train.epochs=1

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="train_model" \
        experiment.model="gmc_test" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=True \
        model.model="gmc_test" \
        model_train.epochs=1 \
        down_train.epochs=1

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" \
        experiment.model="gmc_test" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=True \
        model.model="gmc_test" \
        model_train.epochs=1 \
        down_train.epochs=1

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" \
        experiment.model="gmc_test" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=True \
        experiment.evaluation_mods=[0,1,2,3] \
        model.model="gmc_test" \
        model_train.epochs=1 \
        down_train.epochs=1