import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

# Code is adapted from Lab 5.
class NNPolicy(nn.Module):
    def __init__(self, input_size, output_size, is_multilayer=True,num_hidden=128):
        nn.Module.__init__(self)

        # Switch becomes False when we run GridWorld env.
        self.is_multilayer = is_multilayer
        # When single layer net, need self.l1 to output matrix in output_size
        if not is_multilayer:
            num_hidden = output_size

        self.l1 = nn.Linear(input_size, num_hidden)
        if is_multilayer:
            self.l2 = nn.Linear(num_hidden, output_size)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Args:
            x: input tensor (first dimension is a batch dimension)

        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE
        output = self.l1(x)
        if self.is_multilayer:
            output = F.relu(output)
            output = self.l2(output)
        output = F.softmax(output, dim=-1)
        return output

    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains
        a probability of perfoming corresponding action in all states (one for every state action pair).

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        # YOUR CODE HERE
        output = self.forward(obs)
        action_probs = output.gather(dim=-1, index=actions)
        return action_probs

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        output = self.forward(obs)
        # Convert into tensor.
        actions = torch.ShortTensor(np.arange(0, output.size(0)))
        # Use torch.multinomial to sample action given the action probs.
        idx = output.multinomial(num_samples=1, replacement=True)
        action = actions[idx].item()
        return action
