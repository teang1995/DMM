for seed in 1000 1001 1002 1003 1004
do

root_name="final-ac-base-2--"
wandb_name=$root_name$seed
/data/data_goqhadl92998/miniconda3/envs/coot/bin/python train_retrieval.py \
-c config/retrieval/paper2020/anet_coot.yaml \
--seed=$seed \
--is_baseline=True \
-r $wandb_name

done
