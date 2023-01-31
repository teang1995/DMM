for seed in 1000
do

root_name="low-ac-ours-4-"
wandb_name=$root_name$seed
CUDA_VISIBLE_DEVICES=5 /data/data_goqhadl92998/miniconda3/envs/coot/bin/python train_retrieval.py \
-c config/retrieval/paper2020/anet_coot.yaml \
--seed=$seed \
--is_baseline=False \
-r $wandb_name

done
