import numpy as np
import itertools as it
from model.rl_agent import RLOnAcceptanceAgent
from hyperopt import fmin, tpe, hp


class Economy(object):

    def __init__(self, repartition_of_roles, t_max, agent_model, economy_model,
                 cognitive_parameters=None):

        self.t_max = t_max
        self.cognitive_parameters = cognitive_parameters
        self.agent_model = agent_model
        self.repartition_of_roles = np.asarray(repartition_of_roles)

        self.n_goods = len(self.repartition_of_roles)
        self.roles = self.get_roles(self.n_goods, economy_model)

        self.n_agent = sum(self.repartition_of_roles)

        self.markets = self.get_markets(self.n_goods)
        self.exchanges_types = list(it.combinations(range(self.n_goods), r=2))

        self.agents = self.create_agents()

    @staticmethod
    def get_markets(n_goods):
        markets = {}
        for i in it.permutations(range(n_goods), r=2):
            markets[i] = []
        return markets

    @staticmethod
    def get_roles(n_goods, model):

        roles = np.zeros((n_goods, 2), dtype=int)
        if model == "prod: i+1":
            for i in range(n_goods):
                roles[i] = (i+1) % n_goods, i

        elif model == "prod: i-1":
            for i in range(n_goods):
                roles[i] = (i-1) % n_goods, i

        else:
            raise Exception("Model '{}' is not defined.".format(model))

        return roles

    def create_agents(self):

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.repartition_of_roles):

            i, j = self.roles[agent_type]

            for ind in range(n):
                a = self.agent_model(
                    prod=i, cons=j,
                    cognitive_parameters=self.cognitive_parameters,
                    n_goods=self.n_goods,
                    idx=agent_idx)

                agents.append(a)
                agent_idx += 1

        return agents

    def run(self):

        self.agents = self.create_agents()

        for t in range(self.t_max):

            self.time_step()

    def time_step(self):

        # ---------- MANAGE EXCHANGES ----- #
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

        success_idx = []
        for i, j in self.exchanges_types:

            a1 = self.markets[(i, j)]
            a2 = self.markets[(j, i)]
            min_a = int(min([len(a1), len(a2)]))

            if min_a:

                success_idx += list(np.random.choice(a1, size=min_a))
                success_idx += list(np.random.choice(a2, size=min_a))

        for idx in success_idx:

            agent = self.agents[idx]
            agent.proceed_to_exchange()


def fun_3_goods(args):

    t_max = 100

    repartition_of_roles = int(args[0]), int(args[1]), int(args[2])

    cognitive_parameters = {
        "alpha": 0.1,
        "temp": 0.01
    }

    parameters = {
        "repartition_of_roles": repartition_of_roles,
        "economy_model": "prod: i+1",
        "agent_model": RLOnAcceptanceAgent,
        "cognitive_parameters": cognitive_parameters,
        "t_max": t_max
    }

    frequencies = []
    e = Economy(**parameters)
    for t in range(t_max):
        e.time_step()
        if t >= int(0.5 * t_max):
            # We look for m = 0, so the market (1, 2) should be empty
            f = len(e.markets[(1, 2)]) + len(e.markets[(2, 1)])
            frequencies.append(f)

    return np.mean(frequencies)


def optimize_3_goods():

    space = [
        hp.quniform('a', 10, 50, 1),
        hp.quniform('b', 10, 50, 1),
        hp.quniform('c', 10, 50, 1)
    ]

    best = fmin(
        fn=fun_3_goods,
        space=space,
        algo=tpe.suggest,
        max_evals=50
    )
    print("Best repartition found:", best["a"], best["b"], best["c"])


def fun_4_goods(args):

    t_max = 100

    repartition_of_roles = int(args[0]), int(args[1]), int(args[2]), int(args[3])

    cognitive_parameters = {
        "alpha": 0.1,
        "temp": 0.01
    }

    parameters = {
        "repartition_of_roles": repartition_of_roles,
        "economy_model": "prod: i-1",
        "agent_model": RLOnAcceptanceAgent,
        "cognitive_parameters": cognitive_parameters,
        "t_max": t_max
    }

    frequencies = []
    e = Economy(**parameters)
    for t in range(t_max):
        e.time_step()
        if t >= int(0.5 * t_max):
            # We look for m = 0, so all markets where 0 doesn't circulate should be empty.
            f = len(e.markets[(1, 2)]) + len(e.markets[(2, 1)]) + \
                len(e.markets[(1, 3)]) + len(e.markets[(3, 1)]) + \
                len(e.markets[(2, 3)]) + len(e.markets[(3, 2)])
            frequencies.append(f)

    return np.mean(frequencies)


def optimize_4_goods():

    space = [
        hp.quniform('a', 10, 50, 1),
        hp.quniform('b', 10, 50, 1),
        hp.quniform('c', 10, 50, 1),
        hp.quniform('d', 10, 50, 1)
    ]

    best = fmin(
        fn=fun_4_goods,
        space=space,
        algo=tpe.suggest,
        max_evals=50
    )
    print("Best repartition found:", best["a"], best["b"], best["c"], best["d"])


def fun_5_goods(args):

    t_max = 100

    repartition_of_roles = int(args[0]), int(args[1]), int(args[2]), int(args[3]), int(args[4])

    cognitive_parameters = {
        "alpha": 0.1,
        "temp": 0.01
    }

    parameters = {
        "repartition_of_roles": repartition_of_roles,
        "economy_model": "prod: i-1",
        "agent_model": RLOnAcceptanceAgent,
        "cognitive_parameters": cognitive_parameters,
        "t_max": t_max
    }

    frequencies = []
    e = Economy(**parameters)
    for t in range(t_max):
        e.time_step()
        if t >= int(0.5 * t_max):
            # We look for m = 0, so all markets where 0 doesn't circulate should be empty.
            f = len(e.markets[(1, 2)]) + len(e.markets[(2, 1)]) + \
                len(e.markets[(1, 3)]) + len(e.markets[(3, 1)]) + \
                len(e.markets[(2, 3)]) + len(e.markets[(3, 2)]) + \
                len(e.markets[(1, 4)]) + len(e.markets[(4, 1)]) + \
                len(e.markets[(2, 4)]) + len(e.markets[(4, 2)]) + \
                len(e.markets[(3, 4)]) + len(e.markets[(4, 3)])
            frequencies.append(f)

    return np.mean(frequencies)


def optimize_5_goods():

    space = [
        hp.quniform('a', 30, 100, 1),
        hp.quniform('b', 30, 100, 1),
        hp.quniform('c', 30, 100, 1),
        hp.quniform('d', 30, 100, 1),
        hp.quniform('e', 30, 100, 1),
    ]

    best = fmin(
        fn=fun_5_goods,
        space=space,
        algo=tpe.suggest,
        max_evals=300
    )
    print("Best repartition found: {}, {}, {}, {}, {}".format(
        int(best["a"]), int(best["b"]), int(best["c"]), int(best["d"]), int(best["e"])))

if __name__ == "__main__":

    optimize_3_goods()
