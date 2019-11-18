import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.no_of_current_transition_samples = 0
    
    def add_transition_to_replay_buffer(self, transition):
        self.no_of_current_transition_samples +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample_a_transition_from_replay_buffer(self, batch_size):
        # delete 1/5th of the initial transitions from the buffer when full
        if self.no_of_current_transition_samples > self.max_size:
            del self.buffer[0:int(self.no_of_current_transition_samples / 5)]
            self.no_of_current_transition_samples = len(self.buffer)

        '''Randomly sample batch_size many samples from the replay buffer'''
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for idx in indices:
            s, a, r, s_, d = self.buffer[idx]
            state.append(np.array(s, copy=False)) #instantiating a numpy array with copy = False creates a new copy, just like python's default deepcopy()
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
