CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosi" \
        experiment.model="gmc24" \
        model.model='gmc24' \
        experiment.stage="train_model" \
        model_train.batch_size=24 \
        down_train.batch_size=24

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosi" \
        experiment.model="gmc24" \
        model.model='gmc24' \
        experiment.evaluation_mods=[0] \
        experiment.stage="evaluate_downstream_classifier" \
        model_train.batch_size=24 \
        down_train.batch_size=24

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosi" \
        experiment.model="gmc24" \
        model.model='gmc24' \
        experiment.evaluation_mods=[1] \
        experiment.stage="evaluate_downstream_classifier" \
        model_train.batch_size=24 \
        down_train.batch_size=24

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosi" \
        experiment.model="gmc24" \
        model.model='gmc24' \
        experiment.evaluation_mods=[2] \
        experiment.stage="evaluate_downstream_classifier" \
        model_train.batch_size=24 \
        down_train.batch_size=24

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosi" \
        experiment.model="gmc24" \
        model.model='gmc24' \
        experiment.evaluation_mods=[0,1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        model_train.batch_size=24 \
        down_train.batch_size=24

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosi" \
        experiment.model="gmc24" \
        model.model='gmc24' \
        experiment.evaluation_mods=[0,1] \
        experiment.stage="evaluate_downstream_classifier" \
        model_train.batch_size=24 \
        down_train.batch_size=24

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosi" \
        experiment.model="gmc24" \
        model.model='gmc24' \
        experiment.evaluation_mods=[1,2] \
        experiment.stage="evaluate_downstream_classifier" \
        model_train.batch_size=24 \
        down_train.batch_size=24

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosi" \
        experiment.model="gmc24" \
        model.model='gmc24' \
        experiment.evaluation_mods=[0,2] \
        experiment.stage="evaluate_downstream_classifier" \
        model_train.batch_size=24 \
        down_train.batch_size=24
