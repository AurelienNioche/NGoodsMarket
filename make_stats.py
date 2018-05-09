import main
from analysis import data_format, stats


def run(agent_model='RLAgent', fake=True):

    to_compare = []
    names = []

    for n in 3, 4:

        bkp = {}

        for eq in True, False:
            bkp[f'single_{n}_{"equal" if eq else "not_equal"}'] = \
                main.get_single_data(n_good=n, equal_repartition=eq, agent_model=agent_model, fake=fake)

        names.append(f'g={n}')
        data = data_format.for_stats(*bkp.values())
        to_compare.append(data)

    stats.run(to_compare=to_compare, names=names)


if __name__ == "__main__":
    run()