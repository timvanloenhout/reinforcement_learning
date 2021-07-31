"""This script produces all visuals concerning the gradients.\n
This thus makes the line plots for gradient variance, gradient kurthosis and
gradient sign ratio for all models, for all environments. It will save them
in the same folder from where this script is called.\n\n

IN CASE OF FILENOTEXIST ERRORS:
you can manually set the ROOT variable to a folder called 'policy gradients'
\n\n

You can change for which runs the graphs are made by altering the
ENVIRONMENT and TRAINING_MODELS global variables at the top of this file.
"""

import numpy as np
import pickle as pkl
from scipy.stats import moment
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')


# These globals determine for which run the saved gradients are put into graphs.
ENVIRONMENT = "GridWorld"  # use either Gridworld or CartPole
TRAINING_MODEL = "baseline_gpomdp"  # use either baseline_gpomdp or policies
# This path points to the folder from where the saved gradients are obtained.
ROOT = os.path.join('..', 'train_with_' + TRAINING_MODEL,
                    'outputs_' + ENVIRONMENT, 'policy_gradients')
HIDDEN_LAYERS = 128  # Don't touch this one


def create_plot(stats, env, stat):
    df = pd.DataFrame(columns=['algorithm','episode_number',stat])
    i = 0
    for key in stats.keys():
        if env in key:
            # Get algorithm name, episode_nr and stat value from key
            key_split = key.split(env)
            alg = key_split[0][:-1]
            alg = alg.split('_baseline_')
            if 'None' in alg:
                alg_name = alg[0].upper()
            else:
                alg_name = 'whitened GPOMDP'
            freeze = int(key_split[1][8:])
            values = stats[key][stat]

            # loop over all seeds in values and add row to dataframe
            for value in values:
                # Add tot dataframe
                df.loc[i] = [alg_name, freeze, value]
                i += 1

    sns.lineplot(data=df, x="episode_number", y=stat, hue="algorithm")
    plt.legend(loc='best')
    plt.title(f'{stat} of the model gradients on {env}')

    file_name = env + '_' + stat
    plt.savefig(file_name)
    plt.clf()


def scale(data, min, max):
    nom = (data-data.min(axis=0))*2
    denom = data.max(axis=0) - data.min(axis=0)
    denom[denom==0] = 1
    return -1 + nom/denom


def gen_gradients_files(root):
    '''Yield all files and configurations containting policy gradients from
    root.
    '''

    # Where to load data and save generated figures.
    for model_folder in os.scandir(root):
        if model_folder.is_file():
            continue
        # Note the model name and baseline from the folder name.
        policy = 'reinforce'
        baseline = None
        if model_folder.name.find('gpomdp') >= 0:
            policy = 'gpomdp'
            if model_folder.name.find('normalized') >= 0:
                baseline = 'normalized_baseline'

        # Loop over all different configurations per model + baseline.
        for config_folder in os.scandir(model_folder.path):
            folder_name = config_folder.name.split('_')
            if config_folder.is_file():
                continue
            while 'baseline' in folder_name:
                folder_name.remove('baseline')
            while 'v1' in folder_name:
                folder_name.remove('v1')
            if baseline is not None:
                folder_name = folder_name[1:]

            # Yield the hyperparameters and freeze of the gradients at path.
            for gradients_file in os.scandir(config_folder.path):
                if gradients_file.is_dir():
                    continue
                config = {
                    "environment" : folder_name[4],
                    "policy" : policy,
                    "learning_rate" : float(folder_name[9]),
                    "seed" : int(folder_name[6]),
                    "hidden_layer" : HIDDEN_LAYERS,
                    "sampling_freq": 20,
                    "baseline": baseline
                }
                freeze_num = int(gradients_file.name.split('_')[1])
                yield gradients_file.path, config, freeze_num


root = ROOT
stats = {}
for gradients_file_path, config, freeze in gen_gradients_files(root):

    # Get name for setting + freeze point
    setting_freeze = "{}_baseline_{}_{}_freeze_{}".format(config["policy"],
                                                          config["baseline"],
                                                          config["environment"].replace('-', '_'),
                                                          freeze)
    if setting_freeze not in stats.keys():
        stats[setting_freeze] = {'variance':[], 'kurtosis':[],
                                 'positive_sign_ratio':[]}

    # load gradients
    grads = np.load(gradients_file_path, allow_pickle=True)
    grads = grads['arr_0'].item()

    # 500 gradients per parameter theta
    if 'l2.weight' in grads:
        grads = np.hstack((grads['l1.weight'],grads['l2.weight']))
    else:
        grads = grads['l1.weight']
    grads = scale(grads, -1, 1) # scale gradients to -1,1

    # statistics per parameter over the 500 roll-outs
    var = moment(grads, moment=2)
    kurt = moment(grads, moment=4)
    sign = np.sum(np.array(grads) >= 0, axis=0)/grads.shape[0]

    # averaged statistics over all parameters and add to stats dict
    var, kurt, sign = np.mean(var), np.mean(kurt), np.mean(sign)
    stats[setting_freeze]['variance'].append(var)
    stats[setting_freeze]['kurtosis'].append(kurt)
    stats[setting_freeze]['positive_sign_ratio'].append(sign)


pkl.dump(stats, open( "stats.pkl", "wb" ) )


# Plotting statistics
create_plot(stats, ENVIRONMENT, 'variance')
create_plot(stats, ENVIRONMENT, 'kurtosis')
create_plot(stats, ENVIRONMENT, 'positive_sign_ratio')
