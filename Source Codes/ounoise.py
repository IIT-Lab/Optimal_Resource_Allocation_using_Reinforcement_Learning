import numpy as np
class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2): #noise is set to be 10% up or down of the actual action by default value of scale
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu#noise at all action dimensions is zero by default
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x)) #np.random.randn fills from standard normal distribution -1 to 1
        self.state = x + dx
        return self.state * self.scale
