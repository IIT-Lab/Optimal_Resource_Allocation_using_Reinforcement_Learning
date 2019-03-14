import random
class XOR:
    def __init__(self):
        self.states = [[1,1], [1,0], [0,1], [0,0]]
        self.current_state = []

    def step(self, action):

        if (self.current_state == [1,1] or self.current_state == [0,0]) and action == 0:
            reward = 1
        elif (self.current_state == [1,0] or self.current_state == [0,1]) and action == 1:
            reward = 1
        else:
            reward = 0

        self.current_state = self._get_obs() #setting a new state after the action has been taken in the environment
        done = True #tells whether its time to reset the environment or not, basically tells if the episode ended or not
        return self.current_state, reward, done

    def get_observation(self):
        obs = random.choice(self.states)
        return obs

    def reset(self):
        self.current_state = self.get_observation()