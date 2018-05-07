import numpy as np
import itertools as it
import math
from model.stupid_agent import StupidAgent
from model.get_paths import get_paths
# from model.utils import softmax

np.seterr(all='raise')


class QLearner(StupidAgent):

    name = "QLearner"

    def __init__(self, prod, cons, n_goods, cognitive_parameters, idx):

        super().__init__(prod=prod, cons=cons, n_goods=n_goods, cognitive_parameters=cognitive_parameters, idx=idx)

        self.alpha, self.beta, self.gamma = cognitive_parameters

        self.q = np.zeros((n_goods, n_goods))

        self.reward = np.zeros((n_goods, n_goods))

        self.fill_reward_matrix()

    def fill_reward_matrix(self):

        self.reward[:, self.C] = 1

        for i in range(self.n_goods):
            self.reward[i, i] = -1
        self.reward[self.C, :] = - 1

    def which_exchange_do_you_want_to_try(self):

        pass

    @staticmethod
    def softmax(x, temp):
        try:
            return np.exp(x / temp) / np.sum(np.exp(x / temp))
        except Warning as w:
            print(x, temp)
            raise Exception(f'{w} [x={x}, temp={temp}]')

    def consume(self):

        super().consume()
        self.learn_from_result()

    def learn_from_result(self):

        self.q[self.attempted_exchange[0], self.attempted_exchange[1]] += \
            self.alpha * (
                    self.consumption +
                    self.gamma * np.max(self.q[self.H, self.reward[self.H] != - 1])
                    - self.q[self.attempted_exchange[0], self.attempted_exchange[1]]
            )

