import random

def ind_max(x):
    """
    returns the index that corresponds to the maximum value in array x
    """
    m = max(x)
    return x.index(m)

class EpsilonGreedy():
    """
    epsilon: percentage of time the bandit explores rather than exploits
    counts: number of times each arm has been pulled
    values: average units of reward observed for each arm
    """
    def __init__(self, epsilon, counts=None, values=None):
        if counts is None:
            self.counts = []
        else:
            self.counts = counts
        
        if values is None:
            self.values = []
        else:
            self.values = values

        self.epsilon = epsilon
        
        return

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm(self):
        if random.random() > self.epsilon:
            return ind_max(self.values)
        else:
            return random.randrange(len(self.values))
  
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
    
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return