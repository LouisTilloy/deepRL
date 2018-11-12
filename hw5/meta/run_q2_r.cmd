python train_policy.py pm --exp_name q2_h1_r --history 1 --discount 0.90 -lr 5e-4 -n 60 --recurrent
python train_policy.py pm --exp_name q2_h10_r --history 10 --discount 0.90 -lr 5e-4 -n 60 --recurrent
python train_policy.py pm --exp_name q2_h60_r --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent
python train_policy.py pm --exp_name q2_h120_r --history 120 --discount 0.90 -lr 5e-4 -n 60 --recurrent