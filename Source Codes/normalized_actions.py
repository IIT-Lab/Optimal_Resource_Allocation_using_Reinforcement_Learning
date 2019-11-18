import gym


class NormalizedActions(gym.ActionWrapper):#For MountainCarContinuous, by default the actions are between 1 and -1

    def _action(self, action):#This convention is used for declaring private variables, functions, methods and classes in a module.
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action
