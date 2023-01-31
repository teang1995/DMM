for seed in 1000 1001 1002 1003 1004
do

root_name="final-youcook2-base-"
wandb_name=$root_name$seed
python3 train_retrieval.py -c config/retrieval/paper2020/yc2_2d3d_coot.yaml --seed=$seed -n 10  --is_baseline=True -r $wandb_name

done
