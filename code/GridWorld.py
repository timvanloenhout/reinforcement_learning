import numpy as np
import sys
from gym.envs.toy_text import discrete
import torch.nn.functional as F
import torch

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridworldEnv(discrete.DiscreteEnv): #
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 5x5 grid looks as follows:

    T   o   o  o   o
    -1  o   o  o   o
    o  -1   x  o   o
    o   o   o  -1  o
    o   o   o  T   o

    x is your position and T are the two terminal states. -1 denotes penalty states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[5,5], regular_reward=0.0, penalty=-1.0, final_reward=2.0, max_steps=100):
        """
        This initializes the vars that we'll be using.
        :param shape: Shape of our grid world.
        :param regular_reward: The reward our agent receives if it arrives at a regular, non-terminal state.
        :param penalty: The penalty our agent recieves when it arrives in a bad state.
        :param final_reward: The terminal reward our agent recieves.
        """
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        # Grid World Structure.
        self.shape = shape
        self.nS, self.nA = np.prod(shape), 4
        self.starting_state = 12
        self.next_state = None
        self.terminal_states = [0, self.nS-1]
        self.penalty_states = [5, 11, 18]
        self.max_steps = 100

        # Keep track of where we are.
        self.current_state = self.starting_state

        # Rewards distribution
        self.penalty = penalty
        self.final_reward = final_reward
        self.regular_reward = regular_reward

        self.MAX_Y, self.MAX_X = shape[0], shape[1]
        self.P = {}
        self.grid = np.arange(self.nS).reshape(self.shape)

        # Initial state distribution is uniform
        isd = np.ones(self.nS) / self.nS

        super(GridworldEnv, self).__init__(self.nS, self.nA, self.P, isd)

    # New function! Encode state as one-hot torch tensor.
    def _get_obs(self, state):
        state, nS = torch.LongTensor([state]), np.prod(self.shape)
        state = F.one_hot(input=state, num_classes=nS).float().squeeze(dim=0).numpy().tolist()
        return state

    def reset(self):
        return self._reset()

    def _reset(self):
        # Reset to starting start.
        self.current_state = self.starting_state
        self.num_steps = 0
        return self._get_obs(self.current_state)

    def step(self, action):
        return self._step(action)


    def _step(self, action):
        assert self.action_space.contains(action)
        self.num_steps += 1

        # self.current_state = self.starting_state if self.next_state is None else self.next_state
        # Get coordinates of where we are in the grid given our current state.
        y, x = np.argwhere(self.grid==self.current_state)[0]

        self.P[self.current_state] = {a : [] for a in range(self.nA)}

        is_done = lambda s: s in self.terminal_states
        done = is_done(self.current_state) or self.num_steps >= self.max_steps

        # We've reached the terminal state, so let's reset our self.next_state to None. (Otherwise, we won't restart
        #   at our starting point with the way the code is written.)
        if done:
            self.next_state = None
            reward = self.final_reward
        # Not a terminal state
        else:
            # Figure out agent's future states.
            ns_up = self.current_state if y == 0 else self.current_state - self.MAX_X
            ns_right = self.current_state if x == (self.MAX_X - 1) else self.current_state + 1
            ns_down = self.current_state if y == (self.MAX_Y - 1) else self.current_state + self.MAX_X
            ns_left = self.current_state if x == 0 else self.current_state - 1
            potential_future_states = [ns_up, ns_right, ns_down, ns_left]

            self.next_state = potential_future_states[action]
            # Base reward on future state. Penalize if reward ends up in penalty state.
            reward = self.penalty if self.next_state in self.penalty_states else self.regular_reward

        # Change current state to the next state.
        self.current_state = self.starting_state if self.next_state is None else self.next_state
        return self._get_obs(self.current_state), reward, done, {}


# Unit test taken from lab 1.
if __name__ == "__main__":
    env = GridworldEnv()
    state = env.reset()
    action_sequence = [3,0,3,0,0]
    print("State", state)
    for a in action_sequence:
        state, reward, is_done, _ = env.step(a)
        print("Action", a, "State", state, "reward", reward, "is done", is_done)
