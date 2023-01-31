for lr in 1e-3 5e-4 1e-4
do
for wd in 0.0 1e-3 1e-4
do

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True model.model="gmc_l_lr${lr}_wd${wd}" \
        model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True model.model="gmc_l_lr${lr}_wd${wd}" \
        model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True \
        experiment.evaluation_mods=[0] \
        model.model="gmc_l_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True \
        experiment.evaluation_mods=[1] \
        model.model="gmc_l_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True \
        experiment.evaluation_mods=[2] \
        model.model="gmc_l_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True \
        experiment.evaluation_mods=[0,1] \
        model.model="gmc_l_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True \
        experiment.evaluation_mods=[1,2] \
        model.model="gmc_l_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True \
        experiment.evaluation_mods=[0,2] \
        model.model="gmc_l_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_l_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=True \
        experiment.evaluation_mods=[0,1,2] \
        model.model="gmc_l_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="train_model" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False model.model="gmc_nl_lr${lr}_wd${wd}" \
        model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="train_downstream_classfier" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False model.model="gmc_nl_lr${lr}_wd${wd}" \
        model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False \
        experiment.evaluation_mods=[0] \
        model.model="gmc_nl_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False \
        experiment.evaluation_mods=[1] \
        model.model="gmc_nl_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False \
        experiment.evaluation_mods=[2] \
        model.model="gmc_nl_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False \
        experiment.evaluation_mods=[0,1] \
        model.model="gmc_nl_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False \
        experiment.evaluation_mods=[1,2] \
        model.model="gmc_nl_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False \
        experiment.evaluation_mods=[0,2] \
        model.model="gmc_nl_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

CUDA_VISIBLE_DEVICES=3 python main_unsupervised.py \
-f with experiment.stage="evaluate_downstream_classifier" experiment.model="gmc_nl_lr${lr}_wd${wd}" \
        experiment.seed=0 experiment.log_name="0802_lrwd" experiment.use_label=False \
        experiment.evaluation_mods=[0,1,2] \
        model.model="gmc_nl_lr${lr}_wd${wd}" model_train.epochs=100 down_train.epochs=50 \
        model_train.learning_rate=$lr model_train.weight_decay=$wd

done
done