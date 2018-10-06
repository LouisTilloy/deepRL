#!/usr/bin/env python

"""
Code to run a trained behavioral imitation network in a gym environment
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np
import tf_util
import gym

from train import create_model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('imitation_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument('--num_rollouts', type=int, default=100)
    args = parser.parse_args()

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

    features_ph, labels_ph, dropout_ph, predictions_ph =\
        create_model(n_features, n_labels, 50)

    # create saver to restore model variables
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf_util.initialize()
        print('loading behavior imitation policy')
        saver.restore(sess, "/tmp/{}.ckpt".format(args.imitation_policy_file))
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
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
