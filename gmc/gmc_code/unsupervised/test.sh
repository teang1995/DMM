# 의문. 왜 model_train.epoch=10이 안먹힐까? experiment에서 통재하는 시드만 먹히는 느낌임.
# experiment.model_train_config.epoch ㄱㄱ?

# for seed in 0
# do
# CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
# -f with experiment.stage="train_model" \
#         experiment.model="gmc1" \
#         experiment.seed=$seed \
#         model.model="gmc1" \
#         model_train.epochs=3

# CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
# -f with experiment.stage="train_downstream_classfier" \
#         experiment.model="gmc1" \
#         experiment.seed=$seed \
#         model.model="gmc1" \
#         model_train.epochs=3 \
#         down_train.epochs=3

# CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
# -f with experiment.stage="evaluate_downstream_classifier" \
#         experiment.model="gmc1" \
#         experiment.seed=$seed \
#         experiment.evaluation_mods=[2] \
#         model.model="gmc1"
# done
for seed in 0
do
CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="train_model" \
        experiment.model="mmix1" \
        model.mixup_type="replace" \
        experiment.seed=$seed \
        model.model="mmix1" \
        model_train.epochs=3

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" \
        experiment.model="mmix1" \
        experiment.seed=$seed \
        model.model="mmix1" \
        model.mixup_type="replace" \
        model_train.epochs=3 \
        down_train.epochs=3

CUDA_VISIBLE_DEVICES=7 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" \
        experiment.model="mmix1" \
        model.mixup_type="replace" \
        experiment.seed=$seed \
        experiment.evaluation_mods=[2] \
        model.model="mmix1"
done