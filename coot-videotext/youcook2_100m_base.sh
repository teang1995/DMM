for seed in 1000 1001 1002 1003 1004
do
root_name="test20220620-youcook2-100m-base-"
wandb_name=$root_name$seed
python3 train_retrieval.py -c config/retrieval/paper2020/yc2_100m_coot.yaml --load_model provided_models/yc2_100m_coot.pth --seed=$seed --is_baseline=True -r $wandb_name
done
