ldmm=1.0

for seed in 0
do
for im in 0.1
do
for offset in 4
do
for in in 0
do

# CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="train_model" experiment.evaluation_mods=[-1] \
#         experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
#         model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=1 experiment.seed=$seed \
#         model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=4 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[-1] \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=1 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

done
done
done
done