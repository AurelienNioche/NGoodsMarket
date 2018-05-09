import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import string
from analysis import graph, data_format


import main


def fig_sim(agent_model='RLAgent'):

    bkp = {}
    for n in 3, 4:
        bkp[f'phase_{n}'] = main.get_pool_data(n_good=n, agent_model=agent_model)
        for eq in True, False:
            bkp[f'single_{n}_{"equal" if eq else "not_equal"}'] = \
                main.get_single_data(n_good=n, equal_repartition=eq, agent_model=agent_model)

    fig = plt.figure(figsize=(15, 6), dpi=200)
    fig.subplots_adjust(left=0.05, bottom=0.1, top=0.94, right=0.98)
    gs = grd.GridSpec(ncols=6, nrows=2, width_ratios=[1.3, 1, 1, 1, 1, 0.4], wspace=0.4, hspace=0.3)

    letter = (i.upper() for i in string.ascii_uppercase)

    for i, n in enumerate((3, 4)):

        ax = fig.add_subplot(gs[i, 0])
        data, labels = data_format.for_phase_diagram(bkp[f'phase_{n}'])
        graph.phase_diagram(data=data, labels=labels, n_good=n, ax=ax, letter=next(letter))

        data = data_format.for_monetary_behavior_over_t(bkp[f'single_{n}_not_equal'])
        graph.monetary_behavior_over_t(data=data, fig=fig, subplot_spec=gs[i, 1], letter=next(letter))

        data = data_format.for_medium_over_t(bkp[f'single_{n}_not_equal'])
        graph.medium_over_t(data=data, fig=fig, subplot_spec=gs[i, 2], letter=next(letter))

        data = data_format.for_monetary_behavior_over_t(bkp[f'single_{n}_equal'])
        graph.monetary_behavior_over_t(data=data, fig=fig, subplot_spec=gs[i, 3], letter=next(letter))

        data = data_format.for_medium_over_t(bkp[f'single_{n}_equal'])
        graph.medium_over_t(data=data, fig=fig, subplot_spec=gs[i, 4], letter=next(letter))

    ax = fig.add_subplot(gs[:, 5])

    means, errors = data_format.for_money_bar_plots(
        bkp[f'single_3_not_equal'],
        bkp[f'single_3_equal'],
        bkp[f'single_4_not_equal'],
        bkp[f'single_4_equal'])

    graph.money_bar_plots(means, errors, labels=[
        '3 goods - Non-un. rep.',
        '3 goods - Un. rep.',
        '4 goods - Non-un. rep.',
        '4 goods - Un. rep.'
    ], ax=ax, letter=next(letter))

    ax.set_aspect(aspect=18, anchor='NE')

    ax0 = fig.add_subplot(gs[:, :])
    ax0.set_axis_off()

    ax0.text(
        s='Non-uniform repartition', x=0.4, y=1.05, horizontalalignment='center', verticalalignment='center',
        transform=ax0.transAxes,
        fontsize=15)

    ax0.text(
        s='Uniform repartition', x=0.74, y=1.05, horizontalalignment='center', verticalalignment='center',
        transform=ax0.transAxes,
        fontsize=15)

    ax0.text(
        s='4 goods', x=-0.045, y=0.22, horizontalalignment='center', verticalalignment='center',
        transform=ax0.transAxes, rotation='vertical',
        fontsize=15)

    ax0.text(
        s='3 goods', x=-0.045, y=0.76, horizontalalignment='center', verticalalignment='center',
        transform=ax0.transAxes, rotation='vertical',
        fontsize=15)

    plt.savefig(os.path.expanduser(f'fig/sim_{agent_model}.pdf'))
    plt.show()


def fake_data(agent_model='RLAgent'):

    bkp = {}
    for n in 3, 4:
        for eq in True, False:
            bkp[f'single_{n}_{"equal" if eq else "not_equal"}'] = \
                main.get_single_data(n_good=n, equal_repartition=eq, agent_model=agent_model)

    fig = plt.figure(figsize=(14, 6), dpi=200)
    fig.subplots_adjust(left=0.05, bottom=0.1, top=0.94, right=0.98)
    gs = grd.GridSpec(ncols=5, nrows=2, width_ratios=[1, 1, 1, 1, 0.4], wspace=0.4, hspace=0.3)

    letter = (i.upper() for i in string.ascii_uppercase)

    for i, n in enumerate((3, 4)):

        ax = fig.add_subplot(gs[i, 0])
        data, labels = data_format.for_phase_diagram(bkp[f'phase_{n}'])
        graph.phase_diagram(data=data, labels=labels, n_good=n, ax=ax, letter=next(letter))

        data = data_format.for_monetary_behavior_over_t(bkp[f'single_{n}_not_equal'])
        graph.monetary_behavior_over_t(data=data, fig=fig, subplot_spec=gs[i, 1], letter=next(letter))

        data = data_format.for_medium_over_t(bkp[f'single_{n}_not_equal'])
        graph.medium_over_t(data=data, fig=fig, subplot_spec=gs[i, 2], letter=next(letter))

        data = data_format.for_monetary_behavior_over_t(bkp[f'single_{n}_equal'])
        graph.monetary_behavior_over_t(data=data, fig=fig, subplot_spec=gs[i, 3], letter=next(letter))

        data = data_format.for_medium_over_t(bkp[f'single_{n}_equal'])
        graph.medium_over_t(data=data, fig=fig, subplot_spec=gs[i, 4], letter=next(letter))

    ax = fig.add_subplot(gs[:, 5])

    means, std = data_format.for_money_bar_plots(
        bkp[f'single_3_not_equal'],
        bkp[f'single_3_equal'],
        bkp[f'single_4_not_equal'],
        bkp[f'single_4_equal'])

    graph.money_bar_plots(means, std, labels=[
        '3 goods - Non-un. rep.',
        '3 goods - Un. rep.',
        '4 goods - Non-un. rep.',
        '4 goods - Un. rep.'
    ], ax=ax, letter=next(letter))

    ax.set_aspect(aspect=18, anchor='NE')

    ax0 = fig.add_subplot(gs[:, :])
    ax0.set_axis_off()

    ax0.text(
        s='Non-uniform repartition', x=0.4, y=1.05, horizontalalignment='center', verticalalignment='center',
        transform=ax0.transAxes,
        fontsize=15)

    ax0.text(
        s='Uniform repartition', x=0.74, y=1.05, horizontalalignment='center', verticalalignment='center',
        transform=ax0.transAxes,
        fontsize=15)

    ax0.text(
        s='4 goods', x=-0.045, y=0.22, horizontalalignment='center', verticalalignment='center',
        transform=ax0.transAxes, rotation='vertical',
        fontsize=15)

    ax0.text(
        s='3 goods', x=-0.045, y=0.76, horizontalalignment='center', verticalalignment='center',
        transform=ax0.transAxes, rotation='vertical',
        fontsize=15)

    plt.savefig(os.path.expanduser(f'fig/sim_{agent_model}.pdf'))
    plt.show()


def run():

    for agent_model in ('RLAgent', 'QLearner'):

        fig_sim(agent_model=agent_model)


if __name__ == "__main__":

    run()
