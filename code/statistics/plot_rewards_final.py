""" This is a script to show the rewards different policies accumulated during
evaluation.\n

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os, sys, glob
from collections import namedtuple, defaultdict
sns.set()


# These globals determine for which run the saved rewards are put into graphs.
ENVIRONMENT = "GridWorld"  # use either Gridworld or CartPole
TRAINING_MODEL = "baseline_gpomdp"  # use either baseline_gpomdp or policies
# These are the actual paths were results are obtained and saved.
ROOT = os.path.join('..', 'train_with_' + TRAINING_MODEL,
                    'outputs_' + ENVIRONMENT, 'rewards')
SAVE_PATH = os.path.join('..', 'train_with_' + TRAINING_MODEL,
                         'outputs_' + ENVIRONMENT, 'figures')


# These are ot the globals you're looking for
SAMPLING_FREQ = 20  # This rescales the x-axis.
COLOURS = {  # Determine which colours are used for which policies.
    'reinforce': 'r',
    'gpomdp': 'b',
    'gpomdp_baseline': 'g'
}



# Smoothing function for nicer plots
def smooth(x, N):
    cumsum = np.cumsum(x) # np.insert(x, 0, 0)
    cumsum_late = np.concatenate([np.zeros((N,)+cumsum.shape[1:]), cumsum], axis=0)
    div_facs = np.arange(cumsum.shape[0]) + 1
    div_facs = np.minimum(div_facs, N)
    # return (cumsum[N:] - cumsum[:-N]) / float(N)
    return (cumsum - cumsum_late[:cumsum.shape[0]]) / div_facs


def load_reward_files(root):
    '''Load all reward arrays into a dictionary with configurations as keys.
    Identical configurations with different seeds are appended under the same
    key.
    '''
    # This named tuple will function as a hashable key.
    Configuration = namedtuple('Config', ["environment",
                                          "policy",
                                          "learning_rate",
                                          "discount_factor",
                                          "baseline"])
    config2rewards = defaultdict(list)

    # Loop over all different configurations per model.
    for rewards_file in os.scandir(root):

        filename = rewards_file.name.split('_')
        if rewards_file.is_dir():
            continue

        while 'baseline' in filename:
            filename.remove('baseline')
        while 'v1' in filename:
            filename.remove('v1')

        npz = np.load(rewards_file.path, allow_pickle=True)
        policy2reward = npz[npz.files[0]].item()

        for policy, reward in policy2reward.items():

            baseline = None
            if 'baseline' in policy or 'normalized' in policy:
                baseline = 'baseline'

            config = Configuration(
                environment=     filename[1],
                policy=          policy,
                learning_rate=   float(filename[5]),
                discount_factor= float(filename[8]),
                baseline=        baseline
            )

            config2rewards[config].append(np.array(reward))

    # Have all rewards be arrays instead of lists
    for config, rewards in config2rewards.items():
        config2rewards[config] = np.array(rewards)

    return config2rewards


def pad_rewards_to_array(config2rewards):
    '''Because of the variable run-length per configuration. We need to pad the
    rewards with their last values before making arrays.'''

    for config, rewards in config2rewards.items():

        max_len = 0
        for ep_reward in rewards:
            max_len = max(len(ep_reward), max_len)

        # Pad the episode rewards untill all are of max length.
        rewards_2d = np.empty((len(rewards), max_len))
        for i, ep_rewards in enumerate(rewards):
            last_reward = ep_rewards[-1]
            for _ in range(max_len - len(ep_rewards)):
                ep_rewards.append(last_reward)
            rewards_2d[i] = ep_rewards

        config2rewards[config] = rewards_2d

    return config2rewards


root = ROOT
save_path = SAVE_PATH
if not os.path.exists(save_path):
    os.makedirs(save_path)
config2rewards = load_reward_files(root)

# Use counter just so we can see something while figures are being generated.
counter = 1
# Plot individual seeds
for config, rewards_arr in config2rewards.items():
    config = config._asdict()

    for rewards in rewards_arr:
        print(f"Generating Figure {counter}")
        # Initializing figure.
        fig = plt.figure(1)

        # Reshaping rewards and defining x axis values.
        rewards = np.squeeze(rewards)
        # Only smoothen rewards array if there are too many points.
        if rewards.shape[0] > 1e3:
            smooth_factor = np.floor(rewards.shape[0]/100).astype(int) # Make smoothing factor dynamic.
            rewards = smooth(rewards, N=smooth_factor)
        episodes = np.arange(rewards.shape[0]) * SAMPLING_FREQ

        plt.ylabel('cumulative rewards')
        plt.xlabel('episode number')
        if config['baseline']:
            colour = COLOURS['gpomdp_baseline']
            label = 'whitened gpomdp'
        else:
            colour = COLOURS[config['policy']]
            label = config['policy']
        plt.plot(episodes, rewards, label=label, c=colour)

        # Saving figure.
        policy_description = "{}_baseline_{}_{}_lr_{}_discount_{}_{}.jpg".format(config["policy"],
                                                                                    config["baseline"],
                                                                                    config["environment"],
                                                                                    config["learning_rate"],
                                                                                    config["discount_factor"],
                                                                                    counter)

        fig.savefig(os.path.join(save_path, policy_description), bbox_inches='tight')
        fig.clear()
        counter +=1


# Plot averages
for version in ('gpomdp', 'all'):
    for config, rewards in config2rewards.items():
        config = config._asdict()
        if version == 'gpomdp' and 'gpomdp' not in config['policy']:
            continue

        # Initializing figure.
        print(f"Generating Figure {counter}")
        fig = plt.figure(1)

        # Reshaping rewards and defining x axis values.
        avg_rewards = rewards.mean(0)
        standard_dev = rewards.std(0)

        # Only smoothen rewards and std array if there are too many points.
        if avg_rewards.shape[0] > 1e3:
            smooth_factor = np.floor(avg_rewards.shape[0]/100).astype(int) # Make smoothing factor dynamic.
            avg_rewards = smooth(avg_rewards, N=smooth_factor)
            standard_dev = smooth(standard_dev, N=smooth_factor)
        episodes = np.arange(avg_rewards.shape[0]) * SAMPLING_FREQ

        if config['baseline']:
            colour = COLOURS['gpomdp_baseline']
            label = 'whitened GPOMDP'
        else:
            colour = COLOURS[config['policy']]
            label = config['policy'].upper()
        plt.plot(episodes, avg_rewards, label=label, c=colour)
        # Visualize standard deviation.
        plt.fill_between(episodes,
                        (avg_rewards - standard_dev),
                        (avg_rewards + standard_dev), alpha=0.5, color=colour)

        # Save figure.
        plt.title(f'average performance of the policies on {config["environment"]}')
        plt.ylabel('cumulative rewards')
        plt.xlabel('episode number')
        # Only show legend if we choose to show several results in one plot.
        # plt.legend(bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=len(files_list), borderaxespad=0.)

        policy_description = "{}_baseline_{}_{}_lr_{}_discount_{}.jpg".format(config["policy"],
                                                                            config["baseline"],
                                                                            config["environment"],
                                                                            config["learning_rate"],
                                                                            config["discount_factor"])

        counter += 1

    plt.legend()
    fig.savefig(os.path.join(save_path, version + ".jpg"), bbox_inches='tight')
    fig.clear()
