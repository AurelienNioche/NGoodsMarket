import numpy as np
from pylab import plt
import matplotlib.gridspec as gridspec
import os
# import ternary.heatmapping
# import itertools

import analysis.money


def get_new_fig_name(fig_name):
    
    init_fig_name = fig_name.split(".")[0]
    i = 2
    while os.path.exists(fig_name):
        fig_name = "{}{}.pdf".format(init_fig_name, i)
        i += 1
    
    return fig_name


def _plot_main_fig(medium, exchange, consumption_ratio, repartition, cognitive_parameters):

    t_max = len(medium)
    n_good = len(medium[0])

    # What is common to all subplots
    fig = plt.figure(figsize=(25, 12))
    fig.patch.set_facecolor('white')

    n_lines = 2
    n_columns = 3

    x = np.arange(t_max)

    # First subplot
    ax = plt.subplot(n_lines, n_columns, 1)
    ax.set_title("Proportion of each type of exchange according to time \n")

    type_of_exchanges = sorted([i for i in exchange[0].keys()])
    y = []
    for i in range(len(type_of_exchanges)):
        y.append([])
    for t in range(t_max):
        for exchange_idx in range(len(type_of_exchanges)):
            y[exchange_idx].append(exchange[t][type_of_exchanges[exchange_idx]])

    ax.set_ylim([-0.02, 1.02])

    for exchange_idx in range(len(type_of_exchanges)):
        ax.plot(x, y[exchange_idx], label="Exchange {}".format(type_of_exchanges[exchange_idx]), linewidth=2)

    ax.legend()

    # Second subplot
    ax = plt.subplot(n_lines, n_columns, 2)
    ax.set_title("Consumption average according to time \n")
    ax.plot(x, consumption_ratio, linewidth=2)

    # Third subplot
    ax = plt.subplot(n_lines, n_columns, 3)
    ax.set_title("Total number of exchanges according to time \n")
    ax.plot(x, consumption_ratio, linewidth=2)

    # Fourth subplot
    ax = plt.subplot(n_lines, n_columns, 4)
    ax.set_title("How many times a good $i$ is used as a mean of exchange \n")

    for i in range(n_good):
        ax.plot(x, [j[i] for j in medium],
                label="Good {}".format(i), linewidth=2)

    ax.legend()

    # Sixth subplot
    ax = plt.subplot(n_lines, n_columns, 5)
    ax.set_title("Parameters")
    ax.axis('off')

    msg = \
        f"Cognitive parameters: {cognitive_parameters}; \n \n " \
        f"Repartition of roles: {repartition}; \n \n " \
        f"Time-steps: {t_max}. \n \n"

    ax.text(0.5, 0.5, msg, ha='center', va='center', size=12)

    plt.savefig('fig/main.pdf')

    plt.close()


def _plot_proportions(proportion):

    # Container for proportions of agents having this or that in hand according to their type
    #  - rows: type of agent
    # - columns: type of good

    n_good = len(proportion[0])

    fig = plt.figure(figsize=(25, 12))
    fig.patch.set_facecolor('white')

    n_lines = n_good
    n_columns = 1

    x = np.arange(len(proportion))

    for agent_type in range(n_good):

        # First subplot
        ax = plt.subplot(n_lines, n_columns, agent_type + 1)
        ax.set_title(f"Proportion of agents of type {agent_type} having good i in hand\n")

        y = []
        for i in range(n_good):
            y.append([])

        for proportions_at_t in proportion:
            for good in range(n_good):
                y[good].append(proportions_at_t[agent_type, good])

        ax.set_ylim([-0.02, 1.02])

        for good in range(n_good):
            ax.plot(x, y[good], label="Good {}".format(good), linewidth=2)

        ax.legend()

    plt.tight_layout()

    plt.savefig(fname='fig/proportion.pdf')


def _bar(means, std, labels, title, subplot_spec=None, fig=None):

    if subplot_spec:
        ax = fig.add_subplot(subplot_spec)
    else:
        fig, ax = plt.figure()

    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(length=0)
    ax.set_title(f"$\{title}$", fontsize=20)

    # print(labels)

    # Set x labels
    labels_pos = np.arange(len(labels))
    ax.set_xticklabels(labels)
    ax.set_xticks(labels_pos)

    ax.set_ylim(0, 1)

    # create
    ax.bar(labels_pos, means, yerr=std, edgecolor="white", align="center", color="black")


def _parameters_plot(data, n_good):

    gs = gridspec.GridSpec(1, 3)

    fig = plt.figure(figsize=(13, 8))

    for i, (k, v) in enumerate(data.items()):
        _bar(labels=v[0], means=v[1], std=v[2], subplot_spec=gs[0, i], fig=fig, title=k)

    plt.savefig(f'fig/parameters_{n_good}.pdf')


def _phase_diagram(data, labels, title, n_good):

    fig, ax = plt.subplots()

    im = ax.imshow(data, cmap="binary", origin="lower")#, vmin=0.5)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    ticks = np.arange(len(labels))

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_title(title)

    ax.set_xlabel(f'$x_{n_good-2}$')
    ax.set_ylabel(f'$x_{n_good-1}$')

    fig.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig(f'fig/phase_{n_good}.pdf')


# def _ternary_plot(points, values):
#
#     fig = plt.figure(figsize=(12, 10))
#
#     ax = fig.add_subplot(111)
#
#     # Array of dimension n * 3
#     points = np.array(list(points))
#
#     unq_points = np.unique(points)
#
#     for i, v in enumerate(unq_points):
#         points[points == v] = i
#
#     tup_points = np.zeros(len(points), dtype=object)
#
#     for i, v in enumerate(points):
#         tup_points[i] = tuple(v)
#
#     data = {tup_points[i]: values[i] for i in range(len(values))}
#
#     # data = {(i, j, k): np.random.random() for i, j, k in itertools.product(range(10), repeat=3)}
#     tfg, tax = ternary.figure(ax=ax, scale=len(unq_points) - 1)
#
#     tax.heatmap(data=data, style="triangular")
#     tax.boundary()
#     tax.ticks(clockwise=True, ticks=list(unq_points))
#
#     tax.left_axis_label("$x_1$", fontsize=20)  # , offset=0.16)
#     tax.right_axis_label("$x_2$", fontsize=20)  # , offset=0.16)
#     tax.bottom_axis_label("$x_3$", fontsize=20)  # , offset=0.06)
#
#     tax._redraw_labels()
#
#     ax.set_axis_off()
#     ax.set_aspect(1)
#
#     plt.savefig("fig/ternary.pdf")

# ------------------------------------------------------------------------------------------------- #


def run(bkp):

    print("Beginning analysis...")

    m = 0

    n_good = len(bkp.repartition[0])

    fixed_type_n = bkp.repartition[0][0]
    n = len(bkp.repartition)

    cog_param = np.array(list(bkp.cognitive_parameters))

    money = np.array([analysis.money.run_with_monetary_choice(bkp.choice[i]) for i in range(n)])
    # money = np.array([analysis.money.run_with_exchange(bkp.exchange[i], m=0)for i in range(n)])

    unq_repartition = np.unique(bkp.repartition)

    scores = np.array([np.mean([money[i] for i in range(n) if bkp.repartition[i] == r]) for r in unq_repartition])

    labels = np.unique([i[-1] for i in unq_repartition])
    n_side = len(labels)
    data = scores.reshape(n_side, n_side).T

    if n_good == 3:
        title = f'Money emergence with $x_0 = {fixed_type_n}$ and $m = {m}$'
    elif n_good == 4:
        title = f'Money emergence with $x_0, x_1 = {fixed_type_n}$ and $m = {m}$'
    else:
        title = f'Money emergence with $m = {m}$'

    _phase_diagram(title=title, data=data, labels=labels, n_good=n_good)

    # ----------------------- #

    # print(n_good)
    # if n_good == 4:
    #
    #     a_unq_repartition = np.array(list(unq_repartition))
    #     a_unq_repartition = a_unq_repartition[:, 1:]  # Remove constant
    #
    #     _ternary_plot(points=a_unq_repartition, values=scores)

    # ------------------------- #

    data = {}
    for i, name in enumerate(("alpha", "beta", "epsilon")):
        unq = np.unique(cog_param[:, i])

        d = [money[cog_param[:, i] == j] for j in unq]
        data[name] = ([f'{j:.2f}' for j in unq], [np.mean(j) for j in d], [np.std(j) for j in d])

    _parameters_plot(data=data, n_good=n_good)


def single(bkp):

    _plot_main_fig(
        medium=bkp['medium'],
        exchange=bkp['exchange'],
        consumption_ratio=bkp['consumption_ratio'],
        repartition=bkp['repartition'],
        cognitive_parameters=bkp['cognitive_parameters']
    )

    _plot_proportions(proportion=bkp['proportion'])
