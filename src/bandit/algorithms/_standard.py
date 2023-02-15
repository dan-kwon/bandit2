import random

class EpsilonGreedy():
    """
    epsilon: percentage of time the bandit explores
    n_arms: number of arms for bandit algorithm
    rewards: average units of reward observed for each successful arm pull
    conv_rates: success rates for each arm
    counts: number of times each arm has been pulled
    values: average number of successes observed for each arm (i.e. conversion rate)
    """
    def __init__(self, epsilon, n_arms, rewards, conv_rates=None, counts=None):
        self.epsilon    = epsilon
        self.rewards    = rewards
        self.conv_rates = [0 for i in range(n_arms)] if conv_rates == None else conv_rates
        self.counts     = [0 for i in range(n_arms)] if counts == None else counts
        self.values     = [i*j for i,j in zip(conv_rates,rewards)]
        # raise error if n_arms does not equal number of entries in counts or values
        if ((n_arms != len(self.counts)) or (n_arms != len(self.values)) or (n_arms != len(self.conv_rates))):
            raise ValueError("n_arms does not match the length of counts/values/conv_rates")
        return

    def select_arm(self):
        if random.random() > self.epsilon:
            chosen_arm = ind_max(self.values)
        else:
            chosen_arm = random.randrange(len(self.values))
        return chosen_arm

    def update(self, chosen_arm, success_flag):
        # increments counts for chosen arm
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        # calculate new average reward for chosen arm
        n = self.counts[chosen_arm]
        prev_value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * prev_value + (1 / float(n)) * success_flag * self.rewards[chosen_arm]
        self.values[chosen_arm] = new_value
        return