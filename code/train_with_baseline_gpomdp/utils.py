"""
File of functions we used in lab. Called 'utils', since I couldn't think of anything better and didn't want them in
main script.
"""

import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import sys
import os

# The number of episodes a model is evaluated for calculating the gradients.
EP_DURING_EVAL = 100
# The number of episodes after which a model will stop training if no better
# reward was found.
CONVERGENCE_THRESHOLD = 2000

# Smoothing function for nicer plots
def smooth(x, N):
    cumsum = np.cumsum(x) # np.insert(x, 0, 0)
    cumsum_late = np.concatenate([np.zeros((N,)+cumsum.shape[1:]), cumsum], axis=0)
    div_facs = np.arange(cumsum.shape[0]) + 1
    div_facs = np.minimum(div_facs, N)
    # return (cumsum[N:] - cumsum[:-N]) / float(N)
    return (cumsum - cumsum_late[:cumsum.shape[0]]) / div_facs

def sample_episode(env, policy, device):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []

    done = False
    state = env.reset()
    while not done:
        # Get action using policy.
        action = policy.sample_action(torch.Tensor(state).to(device)) #.item()
        next_state, reward, done, _ = env.step(action)
        # Append to lists
        states.append(state), actions.append(action), rewards.append(reward), dones.append(done)
        # Update to next state.
        state = next_state

    states, actions, rewards = torch.Tensor(states).to(device), \
                               torch.LongTensor(actions).unsqueeze(dim=1).to(device), \
                               torch.Tensor(rewards).unsqueeze(dim=1).to(device)
    dones = torch.Tensor(dones).unsqueeze(dim=1).to(device)
    return states, actions, rewards, dones


def initialize_dirs(dir_paths):
    """
    Baby function that initializes dir if it doesn't already exist.
    :param dir_paths:
    :return:
    """
    for save_folder in dir_paths:
        # Create folder if it doesn't exist.
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)


def compute_reinforce_loss(policy, episode, discount_factor, device, baseline=None):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Same as gpomdp function, only there's a slight difference in loss equation.
    states, actions, rewards, _ = episode
    rewards = rewards.squeeze(dim=-1)
    G = torch.zeros_like(rewards).to(device)
    for t in reversed(range(rewards.shape[0])):
        G[t] = rewards[t] + ((discount_factor * G[t + 1]) if t + 1 < rewards.shape[0] else 0)

    # Also called "whitening" the gradients.
    if baseline == "normalized_baseline":
        G = (G - G.mean()) / G.std()

    action_probs = torch.log(policy.get_probs(states, actions)).squeeze()
    loss = - (action_probs * G[0]).sum()
    return loss, sum(rewards)


def compute_gpomdp_loss(policy, episode, discount_factor, device, baseline=None):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # YOUR CODE HERE
    states, actions, rewards, _ = episode

    # Calculate rewards.
    rewards = rewards.squeeze(dim=-1)
    G = torch.zeros_like(rewards).to(device)
    # Need to calculate loss using formula G_{t+1} = r_t + \gamma G_{t+1}. If statement makes sure that there isn't
    # an error when t=0. Otherwise, we'd get an error since there's no negative time step.
    for t in reversed(range(rewards.shape[0])):
        G[t] = rewards[t] + ((discount_factor * G[t + 1]) if t + 1 < rewards.shape[0] else 0)

    # Also called "whitening" the gradients.
    if baseline == "normalized_baseline":
        epsilon = 1e-5
        G = (G - G.mean()) /(G.std() + epsilon)

    # Calculate loss.
    action_probs = torch.log(policy.get_probs(states, actions)).squeeze()
    loss = - (action_probs * G).sum()
    return loss, sum(rewards)


def eval_policy(policy, env, config, loss_function):

    # Return gradients per episode
    episode_gradients, losses, rewards_list = dict(), list(), list()

    for _ in range(EP_DURING_EVAL):
        episode = sample_episode(env, policy, config['device'])
        policy.zero_grad()  # We need to reset the optimizer gradients for each new run.
        loss, cum_reward = loss_function(policy, episode, config["discount_factor"], config['device'], config["baseline"])
        loss.backward()

        # Save losses as a list.
        losses.append(loss.item()), rewards_list.append(cum_reward.item())

        # Extracting gradients from policy network.
        for name, param in policy.named_parameters():
            if name not in episode_gradients:
                episode_gradients[name] = []
            episode_gradients[name].append(param.grad.cpu().detach().view(-1))

    episode_gradients = {key: torch.stack(episode_gradients[key], dim=0).numpy() for key in episode_gradients}
    average_loss, average_cum_reward = np.asarray(losses).mean(), np.asarray(rewards_list).mean()
    return episode, average_loss, average_cum_reward, episode_gradients


def run_episodes_policy_gradient(policy, env, config):

    # This makes sure that gradients get saved under different name if baseline is used.
    # policy_name = "{}_{}".format(policy_name, baseline) if baseline is not None else policy_name

    # Setting up for training.
    optimizer = optim.Adam(policy.parameters(), config["learning_rate"])

    # With the way the code is implemented, we only care about val reward and val losses for now.
    val_rewards = {policy_name: list() for policy_name in config["policies"]}
    val_losses = {policy_name: list() for policy_name in config["policies"]}

    # Save training rewards/losses (gpomdp + whitening) only. Save if needed for debugging to or writing report.
    model_rewards, model_losses = list(), list()

    save_env_name = config['environment']
    i = save_env_name.find('-')
    if i > -1:
        save_env_name = save_env_name[:i]

    policy.train()
    best_reward = -float('inf')
    best_episode = -1

    # train_loss_function = compute_gpomdp_loss if config["train_with_policie"] == False else
    for i in range(config["num_episodes"]):

        episode = sample_episode(env, policy, config['device'])
        optimizer.zero_grad()  # We need to reset the optimizer gradients for each new run.
        # With the way it's currently coded, we need the same input and outputs for this to work.
        loss, cum_reward = compute_gpomdp_loss(policy, episode, config["discount_factor"], config['device'],
                                               baseline='normalized_baseline')
        model_rewards.append(cum_reward.item()), model_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if cum_reward > best_reward:
            best_reward = cum_reward
            best_episode = i
        elif i - best_episode >= CONVERGENCE_THRESHOLD:
            print(f"convergence is reached at episode {i}")
            break

        # Validating (or "freezing" training of the model).
        if i % config["sampling_freq"] == 0:

            # We validate the policies: 'gpomdp', 'reinforce', and 'gpomdp'+ whitening
            for policy_name in config["policies"]:

                # Define loss function by the policy.
                validate_function = compute_reinforce_loss if "reinforce" in policy_name else compute_gpomdp_loss
                config['baseline'] = 'normalized_baseline' if 'normalized' in policy_name else None

                episode, avg_loss, cum_reward, current_gradients = eval_policy(policy, env, config, validate_function)

                # Save average cum_reward and loss per validation run.
                # Note cum_reward := average cum_reward observed over N runs during validation. Just renamed cum_rewards
                #  so it fits nicely with rest of code.
                val_rewards[policy_name].append(cum_reward)
                val_losses[policy_name].append(avg_loss)

                # Printing something just so we know what's going on.
                print("Episode {0} {3} had an average loss of {1} and lasted for {2} steps. The cumulative reward is {4}"
                      .format(i, round(avg_loss, 4), len(episode[0]), policy_name.upper(), cum_reward))
                # print("{2} Episode {0} had an average loss of {1}"
                #       .format(i, avg_loss, '\033[92m' if len(episode[0]) >= 195 else '\033[99m', policy_name))

                # Saving policy gradients per 'validation' iteration.
                policy_description = "{}_seed_{}_lr_{}_discount_{}_sampling_freq_{}".format(policy_name,
                                                                                            config["baseline"],
                                                                                            config["environment"].replace('-', '_'),
                                                                                            config["seed"],
                                                                                            config["learning_rate"],
                                                                                            config["discount_factor"],
                                                                                            config["sampling_freq"])
                gradients_path = os.path.join('outputs_' + save_env_name, 'policy_gradients', policy_name, policy_description)
                initialize_dirs(dir_paths=[gradients_path])
                np.savez_compressed(os.path.join(gradients_path, "timestep_{}_gradients".format(i)), current_gradients)

    # Saving results.
    # First, save rewards and losses associated with different policies.
    save_paths = [os.path.join('outputs_' + save_env_name, 'rewards'),
                  os.path.join('outputs_' + save_env_name, 'losses')]
    initialize_dirs(dir_paths=save_paths)
    my_results = [val_rewards, val_losses]
    filename = "seed_{}_lr_{}_discount_{}_sampling_freq_{}".format(config["environment"].replace('-', '_'),
                                                                             config["seed"],
                                                                             config["learning_rate"],
                                                                             config["discount_factor"],
                                                                             config["sampling_freq"])

    for save_dir, my_dict in zip(save_paths, my_results):
        np.savez_compressed(os.path.join(save_dir, f"{filename}_rewards"), my_dict)

    # Then save model performance.
    model_performance_path = os.path.join('outputs_' + save_env_name, 'model_performance')
    initialize_dirs(dir_paths=[model_performance_path])
    model_performance = (model_rewards, model_losses)
    np.save(os.path.join(model_performance_path, filename), model_performance)
    return val_rewards, val_losses, model_performance