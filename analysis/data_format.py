import numpy as np
import scipy.stats


def get_money_array(bkp):

    n = len(bkp.repartition)  # Number of economies in this batch

    t_max = len(bkp.monetary_bhv[0][0, :])  # Take first ind from first economy as reference point

    return np.array([np.mean(bkp.monetary_bhv[i][:, int(t_max/2):]) for i in range(n)])


def for_phase_diagram(bkp):

    print("Formating data for phase diagram...")

    n = len(bkp.repartition)  # Number of economies in this batch

    money = get_money_array(bkp)

    unq_repartition = np.unique(bkp.repartition)

    scores = np.array([np.mean([money[i] for i in range(n) if bkp.repartition[i] == r])
                       for r in unq_repartition])
    labels = np.unique([i[-1] for i in unq_repartition])
    n_side = len(labels)
    return scores.reshape(n_side, n_side).T, labels


def for_parameters_plot(bkp):

    cog_param = np.array(list(bkp.cognitive_parameters))

    money = get_money_array(bkp)

    data = {}
    for i, name in enumerate(("alpha", "beta", "epsilon")):
        unq = np.unique(cog_param[:, i])

        d = [money[cog_param[:, i] == j] for j in unq]
        data[name] = ([f'{j:.2f}' for j in unq], [np.mean(j) for j in d], [np.std(j) for j in d])

    return data


def for_monetary_behavior_over_t(bkp):

    n_good = len(bkp['repartition'])
    t_max = len(bkp['monetary_bhv'][0, ])

    agent_type = np.repeat(np.arange(n_good), bkp['repartition'])

    y = np.zeros((n_good, t_max))

    for i in range(n_good):
        for t in range(t_max):
            y[i, t] = np.mean(bkp['monetary_bhv'][agent_type == i, t])

    return y


def for_medium_over_t(bkp):

    n_good = len(bkp['repartition'])

    ref = np.sum(bkp['repartition']) / n_good

    y = bkp['medium'][:, :] / ref

    return y


def for_money_bar_plots(*bkps):

    means = np.zeros(len(bkps))
    sem = np.zeros(len(bkps))

    for i, bkp in enumerate(bkps):

        t_max = len(bkp['monetary_bhv'][0, ])
        money = np.mean(bkp['monetary_bhv'][:, int(t_max / 2):], axis=1)

        means[i] = np.mean(money)
        sem[i] = scipy.stats.sem(money)

    return means, sem


def for_stats(*bkps):

    data = []

    for i, bkp in enumerate(bkps):
        t_max = len(bkp['monetary_bhv'][0, ])
        money = np.mean(bkp['monetary_bhv'][:, int(t_max / 2):], axis=1)
        data.append(money)

    return data
