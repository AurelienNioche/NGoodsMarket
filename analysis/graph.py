import numpy as np
from pylab import plt
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

        # ax.set_ylim([-0.02, 1.02])

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
