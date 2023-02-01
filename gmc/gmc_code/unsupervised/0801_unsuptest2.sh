# for seed in 0 1 2 3 4
# do
# CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
# -f with experiment.stage="train_model" \
#         experiment.model="mmix_test2" \
#         experiment.seed=0 \
#         experiment.log_name="test" \
#         experiment.use_label=False \
#         model.model="mmix_test2" \
#         model.dirichlet_params=[0.1,0.1,0.1] \
#         model.mixup_type="replace" \
#         model_train.epochs=2 \
#         down_train.epochs=2

# CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
# -f with experiment.stage="train_downstream_classfier" \
#         experiment.model="mmix_test2" \
#         experiment.seed=0 \
#         experiment.log_name="test" \
#         experiment.use_label=False \
#         model.model="mmix_test2" \
#         model.dirichlet_params=[0.1,0.1,0.1] \
#         model.mixup_type="replace" \
#         model_train.epochs=2 \
#         down_train.epochs=2

# CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
# -f with experiment.stage="evaluate_downstream_classifier" \
#         experiment.model="mmix_test2" \
#         experiment.seed=0 \
#         experiment.log_name="test" \
#         experiment.use_label=False \
#         experiment.evaluation_mods=[0,1,2] \
#         model.model="mmix_test2" \
#         model.dirichlet_params=[0.1,0.1,0.1] \
#         model.mixup_type="replace" \
#         model_train.epochs=2 \
#         down_train.epochs=2

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="train_model" \
        experiment.model="gmc_test3" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=False \
        model.model="gmc_test3" \
        model_train.epochs=2 \
        down_train.epochs=2

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" \
        experiment.model="gmc_test3" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=False \
        model.model="gmc_test3" \
        model_train.epochs=2 \
        down_train.epochs=2

CUDA_VISIBLE_DEVICES=4 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" \
        experiment.model="gmc_test3" \
        experiment.seed=0 \
        experiment.log_name="test" \
        experiment.use_label=False \
        experiment.evaluation_mods=[0,1,2] \
        model.model="gmc_test3" \
        model_train.epochs=2 \
        down_train.epochs=2