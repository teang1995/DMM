ldmm=1.0

for seed in 0
do
for im in 0.2
do
for offset in 12
do
for in in 0
do

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0,1] experiment.wb_proj='1004_GMC_qualitative2' \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" experiment.vis='tsne' \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0,2] experiment.wb_proj='1004_GMC_qualitative2' \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" experiment.vis='non' \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[1,2] experiment.wb_proj='1004_GMC_qualitative2' \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" experiment.vis='tsne' \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0] experiment.wb_proj='1004_GMC_qualitative2' \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" experiment.vis='non' \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[1] experiment.wb_proj='1004_GMC_qualitative2' \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" experiment.vis='tsne' \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[2] experiment.wb_proj='1004_GMC_qualitative2' \
        experiment.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" experiment.vis='non' \
        model.model="dmm_${ldmm}_${im}_${offset}_${in}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=$in model.init_margin=$im model.sche_offset=$offset model.ldmm=$ldmm


done
done
done
done