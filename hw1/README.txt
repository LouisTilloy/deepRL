Instructions to obtain the data of the pdf.
(Some steps can be long since I used 100 rollouts for every simulation, you can
add --num_rollouts {{number of rollouts}} as a parameter to any python script
other than "train.py" if you want them to run faster.
You can also customize the number of train iterations by adding the parameter 
--final_step {{number of iterations}} to the "train.py" python script.)



I) To obtain the different values in question 2.2 table:
In the "hw1/" folder, open a terminal:

1) To generate the training data and get column 1 and 3:
> python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --store_data
> python run_expert.py experts/HalfCheetah-v2.pkl HalfCheetah-v2 --store_data

2) To train the neural networks:
> python train.py --which Humanoid-v2 --store_several_models
> python train.py --which HalfCheetah-v2
(The "--store_several_models" argument is optional for now but is recommanded since required for step II.1)

3) (optional) To erase the network and train a new one:
> python train.py --which Humanoid-v2 --retrain --store_several_models
> python train.py --which HalfCheetah-v2 --retrain
(The "--store_several_models" argument is optional for now but is recommanded since required for step II.1)

4) To generate the data for columns 2 and 4:
> python run_imitation.py Humanoid-v2 Humanoid-v2
> python run_imitation.py HalfCheetah-v2 HalfCheetah-v2



II) To obtain Figure 1 and Figure 2:
In the "hw1/" folder, open a terminal:

1) If you used the "--store_several_models" in I.2) or I.3), skip this step. Else, to store several checkpoints of the model, run:
> python train.py --which Humanoid-v2 --retrain --store_several_models

2) To plot Figure 1 and Figure 2 (and store figures data on disk in "graph_data/"):
> python generate_graph.py Humanoid-v2 Humanoid-v2

3) (optional) If you have already generated the figures once, you can directly plot them using the data in the folder "graph_data/" by running:
> python generate_graph.py Humanoid-v2 Humanoid-v2 --stds_means_from_files



III) To obtain Figure 3:
In the "hw1/" folder, open a terminal:

1) To train the Humanoid with DAgger and store several checkpoints:
> python train.py --which Humanoid-v2 --dagger --store_several_models

2) (optional) To erase the network and train a new one:
> python train.py --which Humanoid-v2 --dagger --retrain --store_several_models

3) To plot Figure 3:
> python generate_graph.py Humanoid-v2_dagger Humanoid-v2

4) (optional) If you have already generated Figure 3, you can directly plot it using the data in the folder "graph_data/" by running:
> python generate_graph.py Humanoid-v2_dagger --stds_means_from_files



NB: 
you can visualize training by running in "hw1/":
> tensorboard --logdir tensorboard/Humanoid-v2
> tensorboard --logdir tensorboard/HalfCheetah-v2
> tensorboard --logdir tensorboard/Humanoid-v2_dagger