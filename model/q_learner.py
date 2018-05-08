import numpy as np
from scipy.special import expit
from model.stupid_agent import StupidAgent

np.seterr(all='raise')


class QLearner(StupidAgent):

    name = "QLearner"

    def __init__(self, prod, cons, n_goods, cognitive_parameters, idx):

        super().__init__(prod=prod, cons=cons, n_goods=n_goods, cognitive_parameters=cognitive_parameters, idx=idx)

        self.alpha, self.temp, self.gamma = cognitive_parameters

        self.q = np.zeros((n_goods, n_goods), dtype=np.float64)

        self.possible = np.zeros((n_goods, n_goods))

        self.fill_possibilities_matrix()

    def fill_possibilities_matrix(self):

        for i in range(self.n_goods):
            self.possible[i, i] = -1
        self.possible[self.C, :] = - 1

    def which_exchange_do_you_want_to_try(self):

        possible = self.possible[self.H] != - 1

        if sum(possible) > 1:
            p = self.softmax(self.q[self.H, possible], temp=self.temp)
            action = np.random.choice(np.arange(self.n_goods)[possible], p=p)

        else:
            action = np.arange(self.n_goods)[possible][0]

        self.attempted_exchange = self.H, action
        return self.attempted_exchange

    @staticmethod
    def softmax(x, temp):

        try:
            return np.exp(x / temp) / np.sum(np.exp(x / temp))
        except (Warning, FloatingPointError) as w:
            print(x, temp)
            raise Exception(f'{w} [x={x}, temp={temp}]')

    def consume(self):

        super().consume()
        self.learn_from_result()

    def learn_from_result(self):

        self.q[self.attempted_exchange[0], self.attempted_exchange[1]] += \
            \
            self.alpha * (

                    int(self.consumption) +
                    self.gamma * np.max(self.q[self.H, self.possible[self.H] != - 1])

                    - self.q[self.attempted_exchange[0], self.attempted_exchange[1]]
            )

        if self.q[self.attempted_exchange[0], self.attempted_exchange[1]] > 1:
            self.q[self.attempted_exchange[0], self.attempted_exchange[1]] = 1
