ldmm=1.0
lm=0.2
offset=12
in=0

for seed in 0 1 2 3 4
do

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,1,2] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,1] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,2] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[1,2] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[1] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[2] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

done