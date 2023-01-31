ldmm=1.0

for seed in 0
do
for im in 0.15 0.2 0.25 0.3 0.35 0.4
do
for offset in 4 6 8 10 12 14
do
for in in 1
do

CUDA_VISIBLE_DEVICES=2 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=2 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[-1] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

done
done
done
done