for seed in 1000 1001 1002 1003 1004
do
root_name="test20220523-youcook2-100m-ours-"
wandb_name=$root_name$seed
python3 train_retrieval.py -c config/retrieval/paper2020/yc2_100m_coot.yaml --seed=$seed --is_baseline=False -r $wandb_name

done
