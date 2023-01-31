for seed in 1000
do

root_name="final-ac-base-2-s"
wandb_name=$root_name$seed
python3 train_retrieval.py \
-c config/retrieval/paper2020/anet_coot.yaml \
--seed=$seed \
--is_baseline=True \
-r $wandb_name \



done
