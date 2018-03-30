import numpy as np
import itertools as it
from model.stupid_agent import StupidAgent
from model.get_paths import get_paths
from model.utils import softmax


class FrequentistAgent(StupidAgent):

    name = "Frequentist Agent"

    def __init__(self, prod, cons, n_goods, cognitive_parameters, idx):

        super().__init__(prod=prod, cons=cons, n_goods=n_goods, cognitive_parameters=cognitive_parameters, idx=idx)

        self.temp = cognitive_parameters["temp"]

        self.memory_span = cognitive_parameters["memory_span"]

        self.acceptance = self.get_acceptance_dic(n_goods)
        self.memory = self.get_memory_dic(n_goods=n_goods)

        self.paths = get_paths(final_node=cons, n_nodes=n_goods)

    @staticmethod
    def get_acceptance_dic(n_goods):

        acceptance = dict()
        for i in it.permutations(range(n_goods), r=2):
            acceptance[i] = 1.

        return acceptance

    @staticmethod
    def get_memory_dic(n_goods):

        memory = dict()
        for i in it.permutations(range(n_goods), r=2):
            memory[i] = []

        return memory

    def which_exchange_do_you_want_to_try(self):

        exchanges = []
        values = []
        for path in self.paths[self.H]:

            num = 0
            for exchange in path:

                easiness = self.acceptance[exchange]
                if easiness:

                    num += 1/easiness

                else:
                    num = 0
                    break

            if num:
                value = 1/num
            else:
                value = 0

            exchanges.append(path[0])
            values.append(value)

        p = softmax(np.array(values), temp=self.temp)
        self.attempted_exchange = exchanges[np.random.choice(range(len(exchanges)), p=p)]

        return self.attempted_exchange

    def consume(self):

        super().consume()
        self.learn_from_result()

    def learn_from_result(self):

        successful = int(self.H != self.attempted_exchange[0])

        self.memory[self.attempted_exchange].append(successful)
        if len(self.memory[self.attempted_exchange]) > self.memory_span:
            self.memory[self.attempted_exchange] = self.memory[self.attempted_exchange][1:]

        self.acceptance[self.attempted_exchange] = np.mean(self.memory[self.attempted_exchange])

