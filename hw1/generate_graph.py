"""
Function to generate the graphs question 2.3 and 3.2
Be sure to have run this line before using this function:
> python train.py --which {{model}} --retrain --store_several_models
"""
#!/usr/bin/env python

import os
import tensorflow as tf
import numpy as np
import tf_util
import gym
import matplotlib.pyplot as plt

from train import create_model

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    # ****************** ARGPARSE ********************************
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('imitation_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument('--num_rollouts', type=int, default=100)
    parser.add_argument('--stds_means_from_file', action="store_true",
                        help="If you want to generate the graph directly from a list of means and stds"
                             "stored in .txt files")
    args = parser.parse_args()

    # ***************** LOAD ENVIRONMENT AND MODEL ***************
    # Get the model hyper parameters
    if args.envname == "Ant-v2":
        n_labels = 8
        n_features = 111
    elif args.envname == "HalfCheetah-v2":
        n_labels = 6
        n_features = 17
    elif args.envname == "Hopper-v2":
        n_labels = 3
        n_features = 11
    elif args.envname == "Humanoid-v2":
        n_labels = 17
        n_features = 376
    elif args.envname == "Reacher-v2":
        n_labels = 2
        n_features = 11
    elif args.envname == "Walker2d-v2":
        n_labels = 6
        n_features = 17
    else:
        raise ValueError("unknown environment")

    # Create the model
    features_ph, labels_ph, dropout_ph, predictions_ph =\
        create_model(n_features, n_labels, 50)

    # Get the list of mses
    with open("graph_data/mses_{}.txt".format(args.imitation_policy_file), "r") as file:
        list_mses = file.readlines()
        list_mses = [float(mse) for mse in list_mses]

    # ************* SIMULATING FOR ALL CHECKPOINTS ***************
    # Get the list of means and stds
    list_checkpoints = [i * 25000 for i in range(0, len(list_mses))]
    if args.stds_means_from_file:
        with open("graph_data/means_{}.txt".format(args.imitation_policy_file), "r") as file:
            list_means = file.readlines()
            list_means = [float(mean) for mean in list_means]
        with open("graph_data/stds_{}.txt".format(args.imitation_policy_file), "r") as file:
            list_stds = file.readlines()
            list_stds = [float(std) for std in list_stds]
    else:
        list_stds = []
        list_means = []
        for checkpoint, mse in zip(list_checkpoints, list_mses):
            print("Model at step {} of training with mse of {}".format(checkpoint, mse))
            with tf.Session() as sess:
                # create saver to restore model variables
                saver = tf.train.Saver()
                tf_util.initialize()
                print('loading behavior imitation policy')
                saver.restore(sess, "/tmp/{}-{}.ckpt".format(args.imitation_policy_file, checkpoint))
                print('loaded')

                import gym
                env = gym.make(args.envname)
                max_steps = args.max_timesteps or env.spec.timestep_limit

                returns = []
                observations = []
                actions = []
                for i in range(args.num_rollouts):
                    print('iter', i)
                    obs = env.reset()
                    done = False
                    totalr = 0.
                    steps = 0
                    while not done:
                        action = sess.run(predictions_ph, feed_dict={features_ph: [obs],
                                                                     labels_ph: [[0] * n_labels],
                                                                     dropout_ph: 0})
                        observations.append(obs)
                        actions.append(action)
                        obs, r, done, _ = env.step(action)
                        totalr += r
                        steps += 1
                        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                        if steps >= max_steps:
                            break
                    returns.append(totalr)

                list_stds.append(np.std(returns))
                list_means.append(np.mean(returns))

        # Store results
        with open("graph_data/means_{}.txt".format(args.imitation_policy_file), "w") as file:
            for mean in list_means:
                file.write("{}\n".format(mean))
        with open("graph_data/stds_{}.txt".format(args.imitation_policy_file), "w") as file:
            for std in list_stds:
                file.write("{}\n".format(std))

    # ******************** PLOT ******************************
    from scipy import stats
    # Draw the graph of the mean/std as a function of the mse
    # The first point is too far away from the other, so we don't consider it for the sake of graph clarity
    plt.figure()
    plt.errorbar(list_mses[1:], list_means[1:], xerr=0, yerr=list_stds[1:], fmt='o')
    slope, intercept, _, p_value, _ = stats.linregress(np.array(list_mses[1:]), np.array(list_means[1:]))
    line_reg = plt.plot(np.linspace(min(list_mses[1:]), max(list_mses[1:]), 50),
                        slope * np.linspace(min(list_mses[1:]), max(list_mses[1:]), 50) + intercept, 'r',
                        label="linear regression (p value: {})".format(round(p_value, 10)))
    plt.yscale("linear")
    plt.legend(line_reg)
    plt.xlabel("Mean squared error")
    plt.ylabel("Distance covered")
    plt.title("Evolution of the {} clone performance (mean distance covered) with the mses"
              .format(args.imitation_policy_file))

    plt.figure()
    plt.errorbar(list_checkpoints[1:], list_means[1:], xerr=0, yerr=list_stds[1:])
    plt.xlabel("Iteration number")
    plt.ylabel("Distance covered")
    plt.title("Evolution of the {} clone performance (mean distance covered) with training iterations"
              .format(args.imitation_policy_file))

    plt.show()


if __name__ == '__main__':
    main()
