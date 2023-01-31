for seed in 0
do

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" model.infer_mixed=True model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.model="mmix_im_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=True \
        experiment.model="mmix_im_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[2] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1,2] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,1] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[1,2] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.evaluation_mods=[0,2] model.dirichlet_params=[0.1,0.1,0.1] \
        experiment.stage="evaluate_downstream_classifier" model.infer_mixed=False \
        experiment.model="mmix_01" model.model="mmix_im_01" model_train.epochs=40 experiment.seed=$seed model.mixup_type="replace"
done