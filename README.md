# rl_reproducibility
RL 2020 Reproducibility Lab by Jochem Holscher, Bella Nicholson, Tim van Loenhout and Bob Borsboom

## train and validate policies
To do a testrun in which the policies are trained and their gradients, rewards and various other data get saved:
```
cd train_with_baseline_gpomdp
```
or
```
cd train_with_policies
```
and
```
python main.py
```
Inside the train_with_[TRAINING_MODEL] folder is a configuration.py which determines which hyperparameters, environments and policies are used. Feel free to alter any of the capitalized variables in top of any configuration.py file.

As you will have noticed, the training code is split into two similar folders. Most of the functionality is the same, for one core difference.
In the train_with_policies folder, the same policy is used for training the parameters as the policy that validates these parameter and saves the gradients, rewards and other data.
In the train_with_baseline_gpomdp folder, the GPOMDP + whitening is used to train the parameters, and these parameters are validated with the labeled policy from which data is saved.
This distinction is made because REINFORCE cannot learn the environments properly, so to still extract useful data we validate it on the parameters of a policy that is known to train well.

## create graphs
To visualize the variance, kurthosis and sign of the gradients. Go back to the root folder and run:
```
cd statistics
python stats.py
```
This will store the graphs in the statistics folder of the TRAINING_MODEL and ENVIRONMENT selected. The TRAIN_METHOD and ENVIRONMENT are adjustable global variables in the top of stats.py

To visualize the rewards the policies accumulated. Go to the root folder and run:
```
cd statistics
python plot_rewards_final.py
```
This will store the graphs by default in 'train_with_[TRAINING_METHOD]/outputs_[ENVIRONMENT]/figures'. Just like stats.py, TRAIN_METHOD and ENVIRONMENT are adjustable global variables in the top of plot_final_rewards.py