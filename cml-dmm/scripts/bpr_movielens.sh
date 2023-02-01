cd ..
for seed in 1000 1001 1002
do
python main.py --loss bpr --model cml --seed $seed --num_neg 1 --dataset movielens
done