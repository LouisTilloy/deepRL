python train_policy.py pm --exp_name q3_h60_r_g0_1 --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent -g 0.1
python train_policy.py pm --exp_name q3_h60_r_g1_0 --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent -g 1
python train_policy.py pm --exp_name q3_h60_r_g5_0 --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent -g 5
python train_policy.py pm --exp_name q3_h60_r_g10_0 --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent -g 10

python train_policy.py pm --exp_name q3_h1_r_g0_1 --history 1 --discount 0.90 -lr 5e-4 -n 60 --recurrent -g 0.1
python train_policy.py pm --exp_name q3_h1_r_g1_0 --history 1 --discount 0.90 -lr 5e-4 -n 60 --recurrent -g 1
python train_policy.py pm --exp_name q3_h1_r_g5_0 --history 1 --discount 0.90 -lr 5e-4 -n 60 --recurrent -g 5
python train_policy.py pm --exp_name q3_h1_r_g10_0 --history 1 --discount 0.90 -lr 5e-4 -n 60 --recurrent -g 10