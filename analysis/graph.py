import numpy as np
from pylab import plt
import matplotlib.gridspec as gridspec
import os


class GraphicDesigner(object):

    def __init__(self, backup, parameters):

        self.exchanges_list = backup["exchanges"]
        self.mean_utility_list = backup["consumption"]
        self.n_exchanges_list = backup["n_exchanges"]
        self.good_used_as_medium = backup["good_used_as_medium"]
        self.proportions = backup["proportions"]

        self.parameters = parameters

        self.n_goods = len(self.good_used_as_medium[0])

        self.main_figure_name = self.get_fig_name(name="NGoodsMarkets_main_fig")
        self.proportions_figure_name = self.get_fig_name(name="NGoodsMarkets_proportions_fig")

    @staticmethod
    def get_fig_name(name, folder=os.path.expanduser("~/Desktop/NGoodsMarketFigs")):

        os.makedirs(folder, exist_ok=True)

        fig_name = "{}/{}.pdf".format(folder, name)

        init_fig_name = fig_name.split(".")[0]
        i = 2
        while os.path.exists(fig_name):
            fig_name = "{}{}.pdf".format(init_fig_name, i)
            i += 1

        return fig_name

    def plot_main_fig(self):

        # What is common to all subplots
        fig = plt.figure(figsize=(25, 12))
        fig.patch.set_facecolor('white')

        n_lines = 2
        n_columns = 3

        x = np.arange(self.parameters["t_max"])

        # First subplot
        ax = plt.subplot(n_lines, n_columns, 1)
        ax.set_title("Proportion of each type of exchange according to time \n")

        type_of_exchanges = sorted([i for i in self.exchanges_list[0].keys()])
        y = []
        for i in range(len(type_of_exchanges)):
            y.append([])
        for t in range(self.parameters["t_max"]):
            for exchange_idx in range(len(type_of_exchanges)):
                y[exchange_idx].append(self.exchanges_list[t][type_of_exchanges[exchange_idx]])

        ax.set_ylim([-0.02, 1.02])

        for exchange_idx in range(len(type_of_exchanges)):
            ax.plot(x, y[exchange_idx], label="Exchange {}".format(type_of_exchanges[exchange_idx]), linewidth=2)

        ax.legend()

        # Second subplot
        ax = plt.subplot(n_lines, n_columns, 2)
        ax.set_title("Consumption average according to time \n")
        ax.plot(x, self.mean_utility_list, linewidth=2)

        # Third subplot
        ax = plt.subplot(n_lines, n_columns, 3)
        ax.set_title("Total number of exchanges according to time \n")
        ax.plot(x, self.n_exchanges_list, linewidth=2)

        # Fourth subplot
        ax = plt.subplot(n_lines, n_columns, 4)
        ax.set_title("How many times a good $i$ is used as a mean of exchange \n")

        for i in range(self.n_goods):
            ax.plot(x, [j[i] for j in self.good_used_as_medium],
                    label="Good {}".format(i), linewidth=2)

        ax.legend()

        # Sixth subplot
        ax = plt.subplot(n_lines, n_columns, 5)
        ax.set_title("Parameters")
        ax.axis('off')

        msg = \
            "Agent model: {}; \n \n" \
            "Cognitive parameters: {}; \n \n" \
            "Repartition of roles: {}; \n \n " \
            "Economy model: {}; \n \n"\
            "Trials: {}. \n \n".format(
                self.parameters["agent_model"].name,
                self.parameters["cognitive_parameters"],
                self.parameters["repartition_of_roles"],
                self.parameters["economy_model"],
                self.parameters["t_max"]
            )

        ax.text(0.5, 0.5, msg, ha='center', va='center', size=12)

        plt.savefig(self.main_figure_name)

        plt.close()

    def plot_proportions(self):

        # Container for proportions of agents having this or that in hand according to their type
        #  - rows: type of agent
        # - columns: type of good

        fig = plt.figure(figsize=(25, 12))
        fig.patch.set_facecolor('white')

        n_lines = self.n_goods
        n_columns = 1

        x = np.arange(len(self.proportions))

        for agent_type in range(self.n_goods):

            # First subplot
            ax = plt.subplot(n_lines, n_columns, agent_type + 1)
            ax.set_title("Proportion of agents of type {} having good i in hand\n".format(agent_type))

            y = []
            for i in range(self.n_goods):
                y.append([])

            for proportions_at_t in self.proportions:
                for good in range(self.n_goods):
                    y[good].append(proportions_at_t[agent_type, good])

            ax.set_ylim([-0.02, 1.02])

            for good in range(self.n_goods):
                ax.plot(x, y[good], label="Good {}".format(good), linewidth=2)

            ax.legend()

        plt.tight_layout()

        plt.savefig(fname=self.proportions_figure_name)


def represent_results(backup, parameters):
    g = GraphicDesigner(backup=backup, parameters=parameters)
    g.plot_main_fig()
    g.plot_proportions()


# ------------------------------------------------------------------------------------------------- #

def phase_diagram(backup):

    fixed_good = 'x0'
    fixed_type_n = backup.repartition[0][0]
    n = len(backup.repartition)
    t_max = len(backup.medium[0])
    n_good = len(backup.repartition[0])

    plot_phase_diagram(
        medium=backup.medium,
        repartition=backup.repartition,
        fixed_good=fixed_good,
        fixed_type_n=fixed_type_n,
        n=n,
        t_max=t_max,
        n_good=n_good
    )

    # plot_cognitive_parameters(
    #     backup=backup,
    #     repartitions=repartitions,
    #     cognitive_parameters=cognitive_parameters,
    #     fixed_good=fixed_good,
    # )


def plot_cognitive_parameters(backup,
                              repartitions,
                              cognitive_parameters,
                              fixed_good,
                              decision_rule):

    cog_params = {}

    for k in ("alpha", "beta", ("epsilon", "temp")[decision_rule == "softmax"]):
        cog_params[k] = get_all_economies_for_a_cog_param(
            backup=backup,
            repartitions=repartitions,
            cog_param=k,
            cognitive_parameters=cognitive_parameters
        )

    gs = gridspec.GridSpec(1, 3)

    fig = plt.figure(figsize=(10, 10))

    for i, (k, v) in enumerate(cog_params.items()):
        plot_bar(v, fixed_good, subplot_spec=gs[0, i], fig=fig, title="tau" if k == "temp" else k)

    plt.show()


def get_all_economies_for_a_cog_param(backup, cog_param, repartitions, cognitive_parameters):

    """
    get a dic with key: all possible values for
    a cognitive parameter
    and value: all economies using this value
    """

    econ_for_param_with_value = {}

    for r in repartitions:

        for c in cognitive_parameters:

            value = backup[r][c]["parameters"]["cognitive_parameters"][cog_param]

            if econ_for_param_with_value.get(value):
                econ_for_param_with_value[value].append(backup[r][c])
            else:
                econ_for_param_with_value[value] = []
                econ_for_param_with_value[value].append(backup[r][c])

    return econ_for_param_with_value


def plot_bar(backup, fixed_good, title, subplot_spec=None, fig=None):

    data = np.zeros(len(backup), dtype=float)

    for i, (k, v) in enumerate(sorted(backup.items())):

        d = np.array([
            is_monetary(
                    backup=b,
                    parameters=b["parameters"],
                    fixed_good=fixed_good
            )
            for b in v
        ])

        data[i] = np.sum(d) / len(d)

    if subplot_spec:
        ax = fig.add_subplot(subplot_spec)
    else:
        fig, ax = plt.figure()

    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(length=0)
    plt.title(f"$\{title}$", fontsize=20)

    # Set x labels
    labels = ["{:.2f}".format(i) for i in sorted(backup.keys())]
    labels_pos = np.arange(len(labels))
    ax.set_xticklabels(labels)
    ax.set_xticks(labels_pos)

    ax.set_ylim(0, 1)

    # create
    ax.bar(labels_pos, data, edgecolor="white", align="center", color="black")


def plot_phase_diagram(medium, repartition, fixed_good, fixed_type_n, n, t_max, n_good):

    money = np.zeros(n)

    for i in range(n):

        money[i] = is_monetary(
            medium=medium[i],
            t_max=t_max,
            fixed_good=fixed_good,
            n_good=n_good
        )

    unique_repartition = np.unique(repartition)

    scores = np.array([np.mean([money[i] for i in range(n) if repartition[i] == r]) for r in unique_repartition])

    n_side = int(np.sqrt(len(unique_repartition)))

    data = scores.reshape(n_side, n_side).T

    fig, ax = plt.subplots()

    im = ax.imshow(data, cmap="binary", origin="lower")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    labels = np.unique([i[1] for i in unique_repartition])

    ticks = np.arange(n_side)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.xlabel("x1")
    plt.ylabel("x2")

    ax.set_title(f"Money emergence with x0 = {fixed_type_n} and good = {fixed_good}")
    fig.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig('fig/phase.pdf')
    plt.show()


def is_monetary(medium, fixed_good, t_max, n_good):

    factor_medium_difference = 2
    threshold_time_duration = 80

    good_count = np.zeros(n_good, dtype=int)

    last_good_in_memory = None

    for t in range(t_max):

        d = np.array(medium[t])
        idx = np.flatnonzero(d == max(d))

        # If there is one max value
        if len(idx) == 1:

            cond0 = last_good_in_memory == idx

            other_good0, other_good1 = d[d != d[idx]]
            cond1 = d[idx] > other_good0 * factor_medium_difference
            cond2 = d[idx] > other_good1 * factor_medium_difference

            if cond0 and cond1 and cond2:
                good_count[idx] += 1
            else:
                good_count[idx] = 0

            last_good_in_memory = idx
        else:
            # reset all goods count if max() returns two values
            good_count[idx] = 0

    cond0 = max(good_count) > threshold_time_duration
    cond1 = np.argmax(good_count) == fixed_good

    return cond0 and cond1
