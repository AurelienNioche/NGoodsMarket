import numpy as np


class Data:

    def __init__(self, n):

        self.exchange = np.zeros(n, dtype=object)
        self.n_exchange = np.zeros(n, dtype=object)
        self.consumption_ratio = np.zeros(n, dtype=object)  # Consumption ratio
        self.medium = np.zeros(n, dtype=object)
        self.proportion = np.zeros(n, dtype=object)

        self.repartition = np.zeros(n, dtype=object)
        self.cognitive_parameters = np.zeros(n, dtype=object)
        self.choice = np.zeros(n, dtype=object)

        self.i = 0

    def append(self, backup, param):

        self.exchange[self.i] = backup['exchange']
        self.n_exchange[self.i] = backup['n_exchange']
        self.consumption_ratio[self.i] = backup['consumption_ratio']
        self.medium[self.i] = backup['medium']
        self.proportion[self.i] = backup['proportion']

        self.repartition[self.i] = param['repartition']
        self.cognitive_parameters[self.i] = param['cognitive_parameters']

        self.choice[self.i] = backup['choice']

        self.i += 1
