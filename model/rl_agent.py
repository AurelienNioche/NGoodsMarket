import numpy as np
import itertools as it
from model.stupid_agent import StupidAgent
from model.get_paths import get_paths
from model.utils import softmax


class RLOnAcceptanceAgent(StupidAgent):

    name = "RLAgent"

    def __init__(self, prod, cons, n_goods, cognitive_parameters, idx):

        super().__init__(prod=prod, cons=cons, n_goods=n_goods, cognitive_parameters=cognitive_parameters, idx=idx)

        self.alpha = cognitive_parameters["alpha"]
        self.beta = cognitive_parameters["beta"]
        self.epsilon = cognitive_parameters.get("epsilon")
        self.temp = cognitive_parameters.get("temp")

        self.acceptance = self.get_acceptance_dic(n_goods)

        self.paths = get_paths(final_node=cons, n_nodes=n_goods)

    @staticmethod
    def get_acceptance_dic(n_goods):

        acceptance = dict()
        for i in it.permutations(range(n_goods), r=2):
            acceptance[i] = 1.

        return acceptance

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
                print(num)
                value = 1 / (1 + self.beta) ** float(num)
            else:
                value = 0

            exchanges.append(path[0])
            values.append(value)

        return self.epsilon_rule(values, exchanges)

    def softmax_rule(self, values, exchanges):

        p = softmax(np.array(values), temp=self.temp)
        self.attempted_exchange = exchanges[np.random.choice(range(len(exchanges)), p=p)]

        return self.attempted_exchange

    def epsilon_rule(self, values, exchanges):

        max_idx = np.argmax(values)

        if np.random.random() < self.epsilon:
            del exchanges[max_idx]
            random_idx = np.random.randint(len(exchanges))
            self.attempted_exchange = exchanges[random_idx]

        else:
            self.attempted_exchange = exchanges[max_idx]

        return self.attempted_exchange

    def consume(self):

        self.learn_from_result()
        super().consume()

    def learn_from_result(self):

        successful = int(self.H != self.attempted_exchange[0])

        self.acceptance[self.attempted_exchange] += \
            self.alpha * (successful - self.acceptance[self.attempted_exchange])
