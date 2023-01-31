CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_01_32" \
        model.model='mmix_01_32' \
        experiment.stage="train_model" \
        model_train.batch_size=32 \
        down_train.batch_size=32 \
        model.mixup_type='replace' \
        model.dirichlet_params=[0.1,0.1,0.1,0.1]

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_01_32" \
        model.model='mmix_01_32' \
        experiment.evaluation_mods=[0] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_01_32" \
        model.model='mmix_01_32' \
        experiment.evaluation_mods=[1] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_01_32" \
        model.model='mmix_01_32' \
        experiment.evaluation_mods=[2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_01_32" \
        model.model='mmix_01_32' \
        experiment.evaluation_mods=[0,1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_01_32" \
        model.model='mmix_01_32' \
        experiment.evaluation_mods=[0,1] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_01_32" \
        model.model='mmix_01_32' \
        experiment.evaluation_mods=[1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_01_32" \
        model.model='mmix_01_32' \
        experiment.evaluation_mods=[0,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_05_32" \
        model.model='mmix_05_32' \
        experiment.stage="train_model" \
        model_train.batch_size=32 \
        down_train.batch_size=32 \
        model.mixup_type='replace' \
        model.dirichlet_params=[0.5,0.5,0.5,0.5]

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_05_32" \
        model.model='mmix_05_32' \
        experiment.evaluation_mods=[0] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_05_32" \
        model.model='mmix_05_32' \
        experiment.evaluation_mods=[1] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_05_32" \
        model.model='mmix_05_32' \
        experiment.evaluation_mods=[2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_05_32" \
        model.model='mmix_05_32' \
        experiment.evaluation_mods=[0,1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_05_32" \
        model.model='mmix_05_32' \
        experiment.evaluation_mods=[0,1] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_05_32" \
        model.model='mmix_05_32' \
        experiment.evaluation_mods=[1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_05_32" \
        model.model='mmix_05_32' \
        experiment.evaluation_mods=[0,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32


CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_10_32" \
        model.model='mmix_10_32' \
        experiment.stage="train_model" \
        model_train.batch_size=32 \
        down_train.batch_size=32 \
        model.mixup_type='replace' \
        model.dirichlet_params=[1.0,1.0,1.0,1.0]

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_10_32" \
        model.model='mmix_10_32' \
        experiment.evaluation_mods=[0] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_10_32" \
        model.model='mmix_10_32' \
        experiment.evaluation_mods=[1] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_10_32" \
        model.model='mmix_10_32' \
        experiment.evaluation_mods=[2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_10_32" \
        model.model='mmix_10_32' \
        experiment.evaluation_mods=[0,1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_10_32" \
        model.model='mmix_10_32' \
        experiment.evaluation_mods=[0,1] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_10_32" \
        model.model='mmix_10_32' \
        experiment.evaluation_mods=[1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_10_32" \
        model.model='mmix_10_32' \
        experiment.evaluation_mods=[0,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_20_32" \
        model.model='mmix_20_32' \
        experiment.stage="train_model" \
        model_train.batch_size=32 \
        down_train.batch_size=32 \
        model.mixup_type='replace' \
        model.dirichlet_params=[2.0,2.0,2.0,2.0]

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_20_32" \
        model.model='mmix_20_32' \
        experiment.evaluation_mods=[0] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_20_32" \
        model.model='mmix_20_32' \
        experiment.evaluation_mods=[1] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_20_32" \
        model.model='mmix_20_32' \
        experiment.evaluation_mods=[2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_20_32" \
        model.model='mmix_20_32' \
        experiment.evaluation_mods=[0,1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_20_32" \
        model.model='mmix_20_32' \
        experiment.evaluation_mods=[0,1] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_20_32" \
        model.model='mmix_20_32' \
        experiment.evaluation_mods=[1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" \
        experiment.model="mmix_20_32" \
        model.model='mmix_20_32' \
        experiment.evaluation_mods=[0,2] \
        experiment.stage="evaluate_downstream_classifier" \
        down_train.batch_size=32