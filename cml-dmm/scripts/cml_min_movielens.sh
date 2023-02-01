cd ..
for seed in 1000 1001 1002 
do

python main.py --loss mintriplet --model cml --loss_type base --margin 1.5 --seed $seed --dataset movielens
done