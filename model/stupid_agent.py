import numpy as np


class StupidAgent(object):
    name = "StupidAgent"

    def __init__(self, prod, cons, n_goods, cognitive_parameters, idx):

        self.P = prod
        self.C = cons
        self.H = self.P

        self.idx = idx
        self.n_goods = n_goods
        self.cognitive_parameters = cognitive_parameters

        self.consumption = 0
        self.attempted_exchange = None

    def which_exchange_do_you_want_to_try(self):

        possible_desires = [i for i in range(self.n_goods) if i != self.H]
        self.attempted_exchange = self.H, np.random.choice(possible_desires)
        return self.attempted_exchange

    def proceed_to_exchange(self):

        self.H = self.attempted_exchange[1]

    def consume(self):

        self.consumption = self.H == self.C

        if self.consumption:
            self.H = self.P

