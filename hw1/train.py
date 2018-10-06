import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import numpy as np
import random
import pickle
import argparse
import shutil
import load_policy
import gym

val_proportion = 0.1


def create_model(n_features, n_labels, n_units):
    # parameters
    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    # create inputs
    features = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, n_labels])
    dropout = tf.placeholder(dtype=tf.float32, shape=())

    # define network
    layer_1 = tf.layers.dense(features, n_units, activation=tf.nn.relu, use_bias=True, kernel_initializer=initializer)
    layer_1 = tf.layers.dropout(layer_1, rate=dropout)
    layer_2 = tf.layers.dense(layer_1, n_units, activation=tf.nn.relu, use_bias=True, kernel_initializer=initializer)
    layer_2 = tf.layers.dropout(layer_2, rate=dropout)
    layer_3 = tf.layers.dense(layer_2, n_units, activation=tf.nn.relu, use_bias=True, kernel_initializer=initializer)
    layer_3 = tf.layers.dropout(layer_3, rate=dropout)
    predictions = tf.layers.dense(layer_3, n_labels, activation=None, use_bias=False, kernel_initializer=initializer)

    return features, labels, dropout, predictions


if __name__ == "__main__":
    # ************** PARSER *******************
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-w', '--which', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help="Size of the batches.")
    parser.add_argument('-d', '--dropout', type=float, default=0.3,
                        help="Dropout rate.")
    parser.add_argument('-s', '--final_step', type=int, default=500000,
                        help='The global maximum number of steps.')
    parser.add_argument('-u', '--hidden_units', type=int, default=50,
                        help='The number of hidden units in the different layers of the neural network.')
    parser.add_argument('--retrain', action="store_true",
                        help="Train the neural network from the beginning (erase the existing one).")
    parser.add_argument("--store_several_models", action="store_true",
                        help="Whether or not checkpoints model should be stored.")
    parser.add_argument("--dagger", action="store_true",
                        help="Use the dagger algorithm")
    args = parser.parse_args()

    # Get data
    with open('expert_data/{}.pkl'.format(args.which), 'rb') as f:
        result = pickle.load(f)
        actions = np.squeeze(result["actions"])
        observations = result["observations"]

    # Shuffle data
    # A random seed is important so that the splitting validation/train is
    # always the same if the training is done in several times.
    random.seed(0)
    shuffled_indices = list(range(len(actions)))
    random.shuffle(shuffled_indices)
    actions = actions[shuffled_indices]
    observations = observations[shuffled_indices]

    # Get parameters from data
    n_examples = len(actions)
    n_features = len(observations[0])
    n_labels = len(actions[0])

    # Separate in train/val datasets
    val_features = observations[:int(val_proportion * n_examples)]
    train_features = observations[int(val_proportion * n_examples):]
    val_labels = np.squeeze(actions[:int(val_proportion * n_examples)])
    train_labels = np.squeeze(actions[int(val_proportion * n_examples):])

    # Create model
    features_ph, labels_ph, dropout_ph, predictions_ph = create_model(n_features, n_labels, args.hidden_units)

    # Create loss
    mse = tf.losses.mean_squared_error(labels_ph, predictions_ph)
    tf.summary.scalar("mse", mse)

    # Create optimizer
    global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                  initializer=tf.zeros_initializer, trainable=False)
    opt = tf.train.AdamOptimizer().minimize(mse, global_step=global_step)

    # Create saver to save model variables
    saver = tf.train.Saver(max_to_keep=30)
    model_name = args.which if not args.dagger else args.which + "_dagger"

    # Load expert policy if using dagger
    if args.dagger:
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy("experts/{}.pkl".format(args.which))
        print('loaded and built')

    with tf.Session() as sess:
        # restore model or destroy existing one
        if not args.retrain:
            try:
                saver.restore(sess, "/tmp/{}.ckpt".format(model_name))
            except ValueError:
                pass
        else:
            if os.path.exists("tensorboard/{}".format(model_name)):
                shutil.rmtree("tensorboard/{}".format(model_name), ignore_errors=True)

        merged = tf.summary.merge_all()
        # Create writers for tensorboard
        train_writer = tf.summary.FileWriter('tensorboard/{}/train'.format(model_name),
                                             sess.graph)
        val_writer = tf.summary.FileWriter('tensorboard/{}/val'.format(model_name),
                                            sess.graph)

        # initialize variables if no model loaded
        sess.run(tf.global_variables_initializer())

        # launch gym environment for dagger:
        if args.dagger:
            env = gym.make(args.which)

        list_mses = []
        training_step = sess.run(global_step)
        while training_step < args.final_step:
            # get a random subset of the training data
            indices = np.random.randint(low=0, high=len(train_labels), size=args.batch_size)
            features_batch = train_features[indices]
            labels_batch = train_labels[indices]

            # run the optimizer and get the mse
            _, mse_run = sess.run([opt, mse], feed_dict={features_ph: features_batch,
                                                         labels_ph: labels_batch,
                                                         dropout_ph: args.dropout})

            if training_step % 5000 == 0:
                # get the summary for the training set:
                train_summary = sess.run(merged, feed_dict={features_ph: features_batch,
                                                            labels_ph: labels_batch,
                                                            dropout_ph: args.dropout})

                # get the mse and summary for the whole validation set
                val_mse_run, val_summary = sess.run([mse, merged], feed_dict={features_ph: val_features,
                                                                              labels_ph: val_labels,
                                                                              dropout_ph: 0})

                train_writer.add_summary(train_summary, training_step)
                val_writer.add_summary(val_summary, training_step)
                print('{0:04d} val mse: {1:.3f}'.format(training_step, val_mse_run))
                print('{0:04d} train mse: {1:.3f}'.format(training_step, mse_run))
                saver.save(sess, '/tmp/{}.ckpt'.format(model_name))

            # store several models for graph generation
            if args.store_several_models and training_step % 25000 == 0:
                saver.save(sess, '/tmp/{}-{}.ckpt'.format(model_name, training_step))
                list_mses.append(val_mse_run)

            # Add observations from current network and actions from expert to training/validation data
            if args.dagger and training_step % 25000 == 0 and training_step >= 50000:
                print("Generating new training data from current model...")
                observations = []
                expert_actions = []
                for i in range(10):  # 10 rollouts of 1000 time-steps
                    print('iter', i)
                    obs = env.reset()
                    done = False
                    steps = 0
                    while not done:
                        action = sess.run(predictions_ph, feed_dict={features_ph: [obs],
                                                                     labels_ph: [[0] * n_labels],
                                                                     dropout_ph: 0})
                        expert_action = policy_fn(obs[None, :])
                        observations.append(obs)
                        expert_actions.append(expert_action)
                        obs, r, done, _ = env.step(action)
                        steps += 1
                        if steps % 100 == 0: print("%i/%i" % (steps, 1000))
                        if steps >= 1000:
                            break

                # Shuffle actions and observations
                shuffled_indices = list(range(len(observations)))
                random.shuffle(shuffled_indices)
                shuffled_observations = np.array(observations)[shuffled_indices]
                shuffled_expert_actions = np.array(expert_actions)[shuffled_indices]

                # Add new data to validation and training sets
                train_features = np.concatenate((
                    train_features,
                    shuffled_observations[int(val_proportion * len(observations)):])
                )
                train_labels = np.concatenate((
                    train_labels,
                    np.squeeze(shuffled_expert_actions[int(val_proportion * len(observations)):]))
                )
                val_features = np.concatenate((
                    val_features,
                    shuffled_observations[0:int(val_proportion * len(observations))])
                )
                val_labels = np.concatenate((
                    val_labels,
                    np.squeeze(shuffled_expert_actions[0:int(val_proportion * len(observations))]))
                )

                print("Generation done")
            training_step += 1

    if args.store_several_models:
        if not os.path.exists("graph_data"):
            os.makedirs("graph_data")
        with open("graph_data/mses_{}.txt".format(model_name), "w") as file:
            for mse in list_mses:
                file.write("{}\n".format(mse))
