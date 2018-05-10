import numpy as np
from pylab import plt
import matplotlib.gridspec as grd
import os

from analysis import data_format


def _bar(means, errors, labels, title, subplot_spec=None, fig=None):

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
    ax.bar(labels_pos, means, yerr=errors, edgecolor="white", align="center", color="black")


def parameters_plot(data, n_good, fig_name):

    gs = grd.GridSpec(1, 3)

    fig = plt.figure(figsize=(13, 8))

    for i, (k, v) in enumerate(data.items()):
        _bar(labels=v[0], means=v[1], std=v[2], subplot_spec=gs[0, i], fig=fig, title=k)

    if 'fig' in locals():
        print('Saving fig.')
        # noinspection PyUnboundLocalVariable
        fig.tight_layout()

        if fig_name is None:
            fig_name = f'fig/parameters_{n_good}.pdf'

        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        plt.savefig(fig_name)


def phase_diagram(data, labels, n_good,  title=None, ax=None, letter=None, n_ticks=3, fig_name=None):

    if ax is None:
        print('No ax given, I will create a fig.')
        fig, ax = plt.subplots()

    im = ax.imshow(data, cmap="binary", origin="lower", vmin=0.0, vmax=1.0)  # , vmin=0.5)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    step = int(len(labels)/n_ticks)
    lab_to_display = labels[::step]

    ax.set_xticklabels(lab_to_display)
    ax.set_yticklabels(lab_to_display)

    ticks = list(range(len(labels)))[::step]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.tick_params(labelsize=8)

    ax.set_xlabel(f'$x_{n_good-2}$')
    ax.set_ylabel(f'$x_{n_good-1}$')

    ax.set_aspect(1)

    if title is not None:
        ax.set_title(title)

    if letter:
        ax.text(
            s=letter, x=-0.1, y=-0.2, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes,
            fontsize=20)

    if 'fig' in locals():
        print('Saving fig.')
        # noinspection PyUnboundLocalVariable
        fig.tight_layout()

        if fig_name is None:
            fig_name = f'fig/phase_{n_good}.pdf'

        os.makedirs(os.path.dirname(fig_name), exist_ok=True)
        plt.savefig(fig_name)


def monetary_behavior_over_t(data, fig, subplot_spec, letter=None):

    n_good = len(data)
    colors = [f'C{i}' for i in range(n_good)]

    gs = grd.GridSpecFromSubplotSpec(subplot_spec=subplot_spec, ncols=1, nrows=n_good)

    for i in range(n_good):

        ax = fig.add_subplot(gs[i, 0])
        ax.plot(data[i], color=colors[i], linewidth=2)
        ax.set_yticks([0, 1])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(data[i]))

        ax.axhline(y=1 / (n_good - 1), linewidth=1, linestyle='--', color='0.5', zorder=-10)

        if i == (n_good - 1):
            ax.set_xlabel('$t$')
            ax.set_xticks([0, len(data[i])])
        else:
            ax.set_xticks([])

        ax.tick_params(labelsize=8)

    ax0 = fig.add_subplot(gs[:, :])
    ax0.set_axis_off()

    ax0.text(s="Monetary behavior", x=-0.15, y=0.5, horizontalalignment='center', verticalalignment='center',
             transform=ax0.transAxes, fontsize=10, rotation='vertical')

    if letter:
        ax0.text(
            s=letter, x=-0.1, y=-0.1, horizontalalignment='center', verticalalignment='center',
            transform=ax0.transAxes,
            fontsize=20)


def medium_over_t(data, fig, subplot_spec, letter=None):

    n_good = len(data)
    colors = [f'C{i+4}' for i in range(n_good)]

    gs = grd.GridSpecFromSubplotSpec(subplot_spec=subplot_spec, ncols=1, nrows=n_good)

    for i in range(n_good):

        ax = fig.add_subplot(gs[i, 0])
        ax.plot(data[i], color=colors[i], linewidth=2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', f'n/{n_good}'])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(data[i]))
        if i == (n_good - 1):
            ax.set_xlabel('$t$')
            ax.set_xticks([0, len(data[i])])

        else:
            ax.set_xticks([])

        ax.tick_params(labelsize=8)

    ax0 = fig.add_subplot(gs[:, :])
    ax0.set_axis_off()

    ax0.text(s="Used as medium", x=-0.2, y=0.5, horizontalalignment='center', verticalalignment='center',
             transform=ax0.transAxes, fontsize=10, rotation='vertical')

    if letter:
        ax0.text(
            s=letter, x=-0.1, y=-0.1, horizontalalignment='center', verticalalignment='center',
            transform=ax0.transAxes,
            fontsize=20)


def money_bar_plots(means, errors, labels, ax=None, letter=None):

    if ax is None:
        print('No ax given, I will create a fig.')
        fig, ax = plt.subplots()

    if letter:
        ax.text(
            s=letter, x=-0.1, y=-0.68, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes,
            fontsize=20)

    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', length=0)

    # print(labels)

    # Set x labels
    labels_pos = np.arange(len(labels))
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_xticks(labels_pos)

    ax.axhline(y=max(means[0:2])+0.05, xmin=0.15, xmax=0.4, color='black')
    ax.text(s='***',
            y=max(means[0:2])+0.075, x=0.5, horizontalalignment='center', verticalalignment='center')

    ax.axhline(y=max(means[2:4]) + 0.06, xmin=0.63, xmax=0.88, color='black')
    ax.text(s='***',
            y=max(means[2:4]) + 0.085, x=2.5, horizontalalignment='center', verticalalignment='center')

    ax.set_ylim(0, 1)
    ax.set_yticks((0, 0.5, 1))

    ax.set_ylabel("Monetary behavior")

    # create
    ax.bar(labels_pos, means, yerr=errors, edgecolor="white", align="center", color="black")

# ------------------------------------------------------------------------------------------------- #


def run(bkp):

    print("Beginning analysis...")

    m = bkp.m[0]  # Take first economy as reference point
    n_good = len(bkp.repartition[0])  # Take first economy as reference point
    constant_x_value = bkp.constant_x_value[0][0]

    assert np.all(np.array(constant_x_value[0]) == constant_x_value[0][0])

    data, labels = data_format.for_phase_diagram(bkp)

    if n_good == 3:
        title = f'Money emergence with $x_0 = {constant_x_value}$ and $m = {m}$'
    elif n_good == 4:
        title = f'Money emergence with $x_0, x_1 = {constant_x_value}$ and $m = {m}$'
    else:
        title = f'Money emergence with $m = {m}$'

    phase_diagram(title=title, data=data, labels=labels, n_good=n_good,
                  fig_name=f'fig/phase_{n_good}_{bkp.agent_model[0]}.pdf')

    data = data_format.for_parameters_plot(bkp)
    parameters_plot(data=data, n_good=n_good,
                    fig_name=f'fig/parameters_{n_good}_{bkp.agent_model[0]}.pdf')


def single(bkp):

    fig = plt.figure()
    gs = grd.GridSpec(nrows=1, ncols=2, wspace=0.6)

    data = data_format.for_monetary_behavior_over_t(bkp)
    monetary_behavior_over_t(data=data, fig=fig, subplot_spec=gs[0, 0])

    data = data_format.for_medium_over_t(bkp)
    medium_over_t(data=data, fig=fig, subplot_spec=gs[0, 1])

    gs.tight_layout(fig)
    plt.show()
