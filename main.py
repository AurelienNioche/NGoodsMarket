import argparse
import itertools
from model.economy import Economy
from tqdm import tqdm
import numpy as np

from analysis import graph
import backup.backup
import backup.structure
import multiprocessing
import os


def get_parameters(
        n_good=3,
        agent_model='RLAgent',
        m=0,
        constant_x_value=50,
        constant_x_index=np.array([0, ]),
        t_max=100,
        economy_model='prod: i-1',
        range_repartition=range(10, 200, 20),
        n_cog_value=3):

    if agent_model == 'RLAgent':
        first_cog_range = np.linspace(0.1, 0.5, n_cog_value)
        second_cog_range = np.linspace(0.75, 1.5, n_cog_value)
        third_cog_range = np.linspace(0.01, 0.15, n_cog_value)

    else:
        first_cog_range = np.linspace(0.1, 0.5, n_cog_value)
        second_cog_range = np.linspace(0.01, 0.1, n_cog_value)
        third_cog_range = np.linspace(0.1, 0.5, n_cog_value)

    # ------------------------------ #

    repartition = list(itertools.product(range_repartition, repeat=n_good-len(constant_x_index)))

    # ---------- #

    var_param = itertools.product(first_cog_range, second_cog_range, third_cog_range, repartition)

    # ----------- #

    parameters = []

    for alpha, beta, gamma, rpt in var_param:

        complete_rpt = np.zeros(n_good, dtype=int)
        gen_rpt = (i for i in rpt)
        for i in range(n_good):
            if i in constant_x_index:
                complete_rpt[i] = constant_x_value
            else:
                complete_rpt[i] = next(gen_rpt)
        complete_rpt = tuple(complete_rpt)

        param = {
            'cognitive_parameters': (alpha, beta, gamma),
            'repartition': complete_rpt,
            't_max': t_max,
            'economy_model': economy_model,
            'agent_model': agent_model,
            'm': m,
            'seed': np.random.randint(2**32-1)
        }
        parameters.append(param)

    return parameters


def _run(param):

    e = Economy(**param)
    return param, e.run()


def _produce_data(n_good, agent_model):

    tqdm.write("Run simulations.")

    param = get_parameters(n_good=n_good, agent_model=agent_model,
                           constant_x_index=np.array([0, ]) if n_good == 3 else np.array([0, 1]))

    max_ = len(param)

    data = backup.structure.Data(n=len(param))

    with multiprocessing.Pool(processes=os.cpu_count()) as p:

        with tqdm(total=max_) as pbar:
            for pr, b in p.imap_unordered(_run, param):
                data.append(backup=b, param=pr)
                pbar.update()

    return data


def get_single_data(n_good, agent_model='RLAgent', force=False, equal_repartition=False, x_ref=10):

    data_file = \
        f'data/single_{n_good}_{"equal" if equal_repartition else "not_equal"}_{agent_model}.p'

    if force or not os.path.exists(data_file):

        x, y = (x_ref, x_ref) if equal_repartition else (x_ref, x_ref*2)
        parameters = {
            'repartition': tuple([x, ] * 2 + [y, ] * (n_good - 2)),
            'economy_model': 'prod: i-1',
            'agent_model': agent_model,
            't_max': 100,
            'm': 0
        }

        if agent_model == 'RLAgent':
            parameters['cognitive_parameters'] = 0.1, 1, 0.1

        else:
            parameters['cognitive_parameters'] = 0.5, 0.05, 0.2

        e = Economy(**parameters)

        bkp = e.run()
        bkp.update(parameters)
        backup.backup.save(obj=bkp, file_name=data_file)

    else:
        bkp = backup.backup.load(data_file)

    return bkp


def get_pool_data(n_good, agent_model='RLAgent', force=False):

    data_file = f'data/phase_{n_good}_{agent_model}.p'

    if force or not os.path.exists(data_file):
        bkp = _produce_data(n_good, agent_model)
        backup.backup.save(obj=bkp, file_name=data_file)

    else:
        bkp = backup.backup.load(data_file)

    return bkp


def main(args):

    if args.n_good < 3:
        raise Exception("Number of goods has to be greater than 3.")

    elif args.demo:
        bkp = get_single_data(n_good=args.n_good, agent_model=args.agent_model, force=args.force)
        graph.single(bkp)

    else:
        bkp = get_pool_data(n_good=args.n_good, agent_model=args.agent_model, force=args.force)
        graph.run(bkp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run money simulations.')

    parser.add_argument('-d', '--demo', action="store_true", default=False,
                        help="Demonstration mode (run a single eco).")

    parser.add_argument('-f', '--force', action="store_true", default=False,
                        help="Force creation of new data.")

    parser.add_argument('-n', '--n_good', action="store", default=3, type=int,
                        help="How many goods do you want (default = 3)?")

    parser.add_argument('-m', '--agent_model', action="store", default='RLAgent', type=str,
                        help="Which model do you want to use (default = 'RLAgent')?")

    parsed_args = parser.parse_args()

    main(parsed_args)
