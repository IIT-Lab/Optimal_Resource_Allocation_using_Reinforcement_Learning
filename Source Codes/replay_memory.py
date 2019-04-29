import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory_list = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory_list) < self.capacity:
            self.memory_list.append(None)
        self.memory_list[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory_list, batch_size)

    def __len__(self):
        return len(self.memory_list)
