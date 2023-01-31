for seed in 0
do
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" model.infer_mixed=True model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.model="mmix_im_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"

CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"

CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=6 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] model.dirichlet_params=[2.0,2.0,2.0] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_20" model.model="mmix_im_20" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
done