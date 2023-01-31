for seed in 0 1 2 3 4
do
for gmc in 0
do

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0,1] experiment.wb_proj='1009_MMIX_qualitative' \
        experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='non' \
        model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
-f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0,2] experiment.wb_proj='1009_MMIX_qualitative' \
        experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='non' \
        model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
        model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[1,2] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='non' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='non' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[1] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='non' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[2] experiment.wb_proj='1009_MMIX_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='non' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0,1] experiment.wb_proj='1004_GMC_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='umap' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0,2] experiment.wb_proj='1004_GMC_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='umap' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[1,2] experiment.wb_proj='1004_GMC_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='umap' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[0] experiment.wb_proj='1004_GMC_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='umap' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[1] experiment.wb_proj='1004_GMC_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='umap' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

# CUDA_VISIBLE_DEVICES=5 python main_supervised.py \
# -f with experiment.scenario="mosei" experiment.stage="evaluate_visualization" experiment.evaluation_mods=[2] experiment.wb_proj='1004_GMC_qualitative' \
#         experiment.model="base_gmc${gmc}_seed${seed}" experiment.vis='umap' \
#         model.model="base_gmc${gmc}_seed${seed}" model_train.epochs=40 experiment.seed=$seed \
#         model.in_norm=0 model.init_margin=1.0 model.sche_offset=1 model.ldmm=0.0 model.gmc=$gmc

done
done