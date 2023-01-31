ldmm=1.0

for seed in 0
do
for im in 0.1 0.2 0.3
do
for offset in 6 10 12
do
for in in 1 0
do

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" \
        experiment.model="dmm_${seed}_${im}_${offset}_${in}" \
        model.model="dmm_${seed}_${im}_${offset}_${in}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[1,2] \
        experiment.model="dmm_${seed}_${im}_${offset}_${in}" \
        model.model="dmm_${seed}_${im}_${offset}_${in}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,2] \
        experiment.model="dmm_${seed}_${im}_${offset}_${in}" \
        model.model="dmm_${seed}_${im}_${offset}_${in}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,1] \
        experiment.model="dmm_${seed}_${im}_${offset}_${in}" \
        model.model="dmm_${seed}_${im}_${offset}_${in}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0,1,2] \
        experiment.model="dmm_${seed}_${im}_${offset}_${in}" \
        model.model="dmm_${seed}_${im}_${offset}_${in}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[0] \
        experiment.model="dmm_${seed}_${im}_${offset}_${in}" \
        model.model="dmm_${seed}_${im}_${offset}_${in}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[1] \
        experiment.model="dmm_${seed}_${im}_${offset}_${in}" \
        model.model="dmm_${seed}_${im}_${offset}_${in}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm
CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[2] \
        experiment.model="dmm_${seed}_${im}_${offset}_${in}" \
        model.model="dmm_${seed}_${im}_${offset}_${in}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

done
done
done
done