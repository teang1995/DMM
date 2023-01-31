for seed in 0 1 2 3 4
do
for gmc in 0 1
do

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="train_model" experiment.wb_proj='0910_GMC_baseline' \
        experiment.model="base_gmc${gmc}_seed${seed}" \
        model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

CUDA_VISIBLE_DEVICES=0 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_downstream_classifier" experiment.evaluation_mods=[-1] experiment.wb_proj='0910_GMC_baseline' \
        experiment.model="base_gmc${gmc}_seed${seed}" \
        model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

done
done