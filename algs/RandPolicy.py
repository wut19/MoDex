import numpy as np

class RandPolicy:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        return self.env.action_space.sample()
    
class RandMoveAveragePolicy:
    def __init__(self, env, alpha=0.8):
        self.env = env
        self.alpha = alpha
        self.action = np.zeros(self.env.action_space.shape)

    def predict(self, obs):
        self.action = self.alpha * self.action + (1 - self.alpha) * self.env.action_space.sample()
        return self.action