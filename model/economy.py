import numpy as np
import itertools as it

from model.rl_agent import RLAgent  # Necessary for the 'eval' of agent creation
from model.q_learner import QLearner


class Economy(object):

    def __init__(self, repartition, t_max, agent_model, economy_model, cognitive_parameters=None, seed=None, m=0,
                 **kwargs):  # Args for analysis

        np.random.seed(seed)
        self.t_max = t_max
        self.cognitive_parameters = cognitive_parameters
        self.agent_model = agent_model
        self.repartition = np.asarray(repartition)
        self.m = m

        self.n_goods = len(self.repartition)
        self.roles = self.get_roles(self.n_goods, economy_model)

        self.n_agent = sum(self.repartition)

        self.agents = self.create_agents()

        self.t = 0

        self.markets = self.get_markets(self.n_goods)
        self.exchange_types = list(it.combinations(range(self.n_goods), r=2))

        # ---- For backup ----- #
        self.bkp_medium = np.zeros((self.n_goods, self.t_max))
        self.bkp_monetary_bhv = np.zeros((self.n_agent, self.t_max))

    @staticmethod
    def get_markets(n_goods):

        markets = {}
        for i in it.permutations(range(n_goods), r=2):
            markets[i] = []
        return markets

    @staticmethod
    def get_roles(n_goods, model):

        roles = np.zeros((n_goods, 2), dtype=int)
        if model == 'prod: i+1':
            for i in range(n_goods):
                roles[i] = (i+1) % n_goods, i

        elif model == 'prod: i-1':
            for i in range(n_goods):
                roles[i] = (i-1) % n_goods, i

        else:
            raise Exception(f'Model "{model}" is not defined.')

        return roles

    def create_agents(self):

        agents = np.zeros(self.n_agent, dtype=object)

        idx = 0

        for agent_type, n in enumerate(self.repartition):

            i, j = self.roles[agent_type]

            for ind in range(n):
                a = eval(self.agent_model)(
                    prod=i, cons=j,
                    cognitive_parameters=self.cognitive_parameters,
                    n_goods=self.n_goods,
                    idx=idx
                )

                agents[idx] = a
                idx += 1

        return agents

    def run(self):

        for t in range(self.t_max):
            self.time_step(t)

        return {'medium': self.bkp_medium, 'monetary_bhv': self.bkp_monetary_bhv}

    def time_step(self, t):

        self.t = t

        self.organize_encounters()

        # Each agent consumes at the end of each round and adapt his behavior (or not).
        for agent in self.agents:
            agent.consume()

    def organize_encounters(self):

        for k in self.markets:
            self.markets[k] = []

        for agent in self.agents:

            agent_choice = agent.which_exchange_do_you_want_to_try()
            self.markets[agent_choice].append(agent.idx)

            if self.m in (agent.P, agent.C):
                monetary_conform = agent_choice == (agent.P, agent.C)

            else:
                monetary_conform = agent_choice in [(agent.P, self.m), (self.m, agent.C)]

            self.bkp_monetary_bhv[agent.idx, self.t] = int(monetary_conform)

        success_idx = []
        for i, j in self.exchange_types:

            a1 = self.markets[(i, j)]
            a2 = self.markets[(j, i)]
            min_a = int(min([len(a1), len(a2)]))

            if min_a:

                success_idx += list(np.random.choice(a1, size=min_a, replace=False))
                success_idx += list(np.random.choice(a2, size=min_a, replace=False))

        for idx in success_idx:

            agent = self.agents[idx]
            agent.proceed_to_exchange()

            # ---- For backup ----- #
            if agent.attempted_exchange[0] != agent.P and agent.attempted_exchange[1] == agent.C:
                self.bkp_medium[agent.attempted_exchange[0], self.t] += 1


def launch(**kwargs):
    e = Economy(**kwargs)
    return e.run()
