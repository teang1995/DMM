cd ..
for seed in 1000 1001 1002
do

python main.py --loss mintriplet --model cml --loss_type dmm --margin 2.0 --seed $seed --dataset amazon_music
done