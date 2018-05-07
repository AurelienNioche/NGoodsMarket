import numpy as np


def for_phase_diagram(bkp):
    print("Formating data for phase diagram...")

    n = len(bkp.repartition)

    t_max = len(bkp.choice[0])

    money = np.array([np.mean(bkp.choice[i][int(t_max/2):]) for i in range(n)])

    unq_repartition = np.unique(bkp.repartition)

    scores = np.array([np.mean([money[i] for i in range(n) if bkp.repartition[i] == r]) for r in unq_repartition])

    labels = np.unique([i[-1] for i in unq_repartition])
    n_side = len(labels)
    return scores.reshape(n_side, n_side).T, labels


def for_monetary_behavior_over_t(bkp):

    n = len(bkp['repartition'])
    t_max = len(bkp['choice'])

    y = np.zeros((n, t_max))

    for t in range(t_max):

        for i in range(n):
            y[i, t] = bkp['choice'][t][i]

    return y


def for_medium_over_t(bkp):

    n = len(bkp['repartition'])
    t_max = len(bkp['medium'])

    ref = np.sum(bkp['repartition']) / n

    y = np.zeros((n, t_max))

    for t in range(t_max):

        for i in range(n):
            y[i, t] = bkp['medium'][t][i] / ref

    return y


def for_money_bar_plots(*bkps):

    means = np.zeros(len(bkps))
    std = np.zeros(len(bkps))

    for i, bkp in enumerate(bkps):

        n = len(bkp['repartition'])
        t_max = len(bkp['choice'])

        money = np.array([np.mean(bkp['choice'][int(t_max / 2):][i]) for i in range(n)])

        means[i] = np.mean(money)
        std = np.std(money)

    return means, std




