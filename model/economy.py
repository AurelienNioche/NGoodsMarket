import numpy as np
import itertools as it


m = 0


class Economy(object):

    def __init__(self, repartition, t_max, agent_model, economy_model, cognitive_parameters=None, seed=None):

        np.random.seed(seed)
        self.t_max = t_max
        self.cognitive_parameters = cognitive_parameters
        self.agent_model = agent_model
        self.repartition = np.asarray(repartition)

        self.n_goods = len(self.repartition)
        self.roles = self.get_roles(self.n_goods, economy_model)

        self.n_agent = sum(self.repartition)

        self.agents = None

        # ----- For backup at t ----- #
        self.exchange = dict()
        for i in it.combinations(range(self.n_goods), r=2):
            self.exchange[i] = 0
        self.n_exchange = 0
        self.consumption = 0
        self.good_used_as_medium = np.zeros(self.n_goods)

        # Container for proportions of agents having this or that in hand according to their type
        #  - rows: type of agent
        # - columns: type of good

        self.proportions = np.zeros((self.n_goods, self.n_goods))

        self.choice = []

        # ---- For final backup ----- #
        self.back_up = {
            'exchange': [],
            'n_exchange': [],
            'consumption_ratio': [],
            'medium': [],
            'proportion': [],
            'choice': []
        }

        self.markets = self.get_markets(self.n_goods)
        self.exchange_types = list(it.combinations(range(self.n_goods), r=2))

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

        agents = []

        agent_idx = 0

        for agent_type, n in enumerate(self.repartition):

            i, j = self.roles[agent_type]

            for ind in range(n):
                a = self.agent_model(
                    prod=i, cons=j,
                    cognitive_parameters=self.cognitive_parameters,
                    n_goods=self.n_goods,
                    idx=agent_idx
                )

                agents.append(a)
                agent_idx += 1

        return agents

    def run(self):

        self.agents = self.create_agents()
        return self.play()

    def play(self):

        for t in range(self.t_max):
            self.time_step()

        return self.back_up

    def time_step(self):

        self.reinitialize_backup_containers()

        self.compute_proportions()

        # ---------- MANAGE EXCHANGES ----- #
        self.organize_encounters()

        # Each agent consumes at the end of each round and adapt his behavior (or not).
        for agent in self.agents:
            agent.consume()

        self.make_a_backup_for_t()

    def organize_encounters(self):

        for k in self.markets:
            self.markets[k] = []

        for agent in self.agents:
            agent_choice = agent.which_exchange_do_you_want_to_try()
            self.markets[agent_choice].append(agent.idx)

            if m in (agent.P, agent.C):
                self.choice.append(int(agent_choice == (agent.P, agent.C)))

            else:
                self.choice.append(int(agent_choice in [(agent.P, m), (m, agent.C)]))

        success_idx = []
        for i, j in self.exchange_types:

            a1 = self.markets[(i, j)]
            a2 = self.markets[(j, i)]
            min_a = int(min([len(a1), len(a2)]))

            if min_a:

                self.exchange[(i, j)] += min_a
                self.n_exchange += min_a
                success_idx += list(np.random.choice(a1, size=min_a, replace=False))
                success_idx += list(np.random.choice(a2, size=min_a, replace=False))

        for idx in success_idx:
            agent = self.agents[idx]
            agent.proceed_to_exchange()
            # --- Stat
            if agent.attempted_exchange[0] != agent.P and agent.attempted_exchange[1] == agent.C:

                self.good_used_as_medium[agent.attempted_exchange[0]] += 1

    def compute_proportions(self):

        # Container for proportions of agents having this or that in hand according to their type
        #  - rows: type of agent
        # - columns: type of good

        for i in self.agents:
            self.proportions[i.C, i.H] += 1  # Type of agent is his consumption good

        for i in range(self.n_goods):
            self.proportions[i] = self.proportions[i] / self.repartition[i]

    def make_a_backup_for_t(self):

        # Keep a trace from utilities
        self.consumption = sum([a.consumption for a in self.agents])/self.n_agent

        # ----- FOR FUTURE BACKUP ----- #

        for key in self.exchange.keys():
            # Avoid division by zero
            if self.n_exchange > 0:
                self.exchange[key] /= self.n_exchange
            else:
                self.exchange[key] = 0

        # For back up
        self.back_up['exchange'].append(self.exchange.copy())
        self.back_up['consumption_ratio'].append(self.consumption)
        self.back_up['n_exchange'].append(self.n_exchange)
        self.back_up['medium'].append(self.good_used_as_medium.copy())
        self.back_up['proportion'].append(self.proportions.copy())
        self.back_up['choice'].append(np.mean(self.choice))

    def reinitialize_backup_containers(self):

        # Containers for future backup
        for k in self.exchange.keys():
            self.exchange[k] = 0
        self.n_exchange = 0
        self.consumption = 0
        self.good_used_as_medium[:] = 0
        self.proportions[:] = 0
        self.choice = []


def launch(**kwargs):
    e = Economy(**kwargs)
    return e.run()
