import argparse
import itertools
from model.economy import Economy
from tqdm import tqdm
import numpy as np

from model.rl_agent import RLOnAcceptanceAgent
from analysis import graph
import backup.backup
import backup.structure
import multiprocessing
import os


def get_parameters(n_good):

    # ----- set these params ------ #
    fixed_x = (50, ) * (1 if n_good == 3 else 2)
    range_repartition = range(10, 200, 1)
    step = 3
    alpha_range = np.linspace(0.1, 0.5, step)
    beta_range = np.linspace(0.75, 1.5, step)
    gamma_range = np.linspace(0.001, 0.15, step)
    t_max = 100
    economy_model = "prod: i-1"
    agent_model = RLOnAcceptanceAgent

    # ------------------------------ #

    repartition = list(itertools.product(range_repartition, repeat=n_good-len(fixed_x)))

    # ---------- #

    var_param = itertools.product(alpha_range, beta_range, gamma_range, repartition)

    # ----------- #

    parameters = []

    for alpha, beta, gamma, rpt in var_param:
        param = {
            "cognitive_parameters": (alpha, beta, gamma),
            "repartition": fixed_x + rpt,
            "t_max": t_max,
            "economy_model": economy_model,
            "agent_model": agent_model,
            "seed": np.random.randint(2**32-1)
        }
        parameters.append(param)

    return parameters


def _run(param):

    e = Economy(**param)
    return param, e.run()


def _produce_data(n_good):

    tqdm.write("Run simulations.")

    param = get_parameters(n_good)

    max_ = len(param)

    data = backup.structure.Data(n=len(param))

    with multiprocessing.Pool(processes=os.cpu_count()) as p:

        with tqdm(total=max_) as pbar:
            for pr, b in p.imap_unordered(_run, param):
                data.append(backup=b, param=pr)
                pbar.update()

    return data


def get_single_data(n_good, force=False, equal_repartition=False):

    data_file = f'data/single_{n_good}_{"equal" if equal_repartition else "not_equal"}.p'

    if force or not os.path.exists(data_file):

        x, y = (10, 10) if equal_repartition else (10, 20)
        parameters = {
            "repartition": [x, ] * 2 + [y, ] * (n_good - 2),
            "economy_model": "prod: i-1",
            "agent_model": RLOnAcceptanceAgent,
            "cognitive_parameters": (0.1, 1, 0.1),
            "t_max": 100,
        }

        e = Economy(**parameters)

        bkp = e.run()
        bkp.update(parameters)
        backup.backup.save(obj=bkp, file_name=data_file)

    else:
        bkp = backup.backup.load(data_file)

    return bkp


def get_pool_data(n_good, force=False):

    data_file = f'data/phase_{n_good}.p'

    if force or not os.path.exists(data_file):
        bkp = _produce_data(n_good)
        backup.backup.save(obj=bkp, file_name=data_file)

    else:
        bkp = backup.backup.load(data_file)

    return bkp


def main(args):

    if args.n_good < 3:
        raise Exception("Number of goods has to be greater than 3.")

    elif args.demo:
        bkp = get_single_data(n_good=args.n_good, force=args.force)
        graph.single(bkp)

    else:
        bkp = get_pool_data(n_good=args.n_good, force=args.force)
        graph.run(bkp)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run money simulations.')

    parser.add_argument('-d', '--demo', action="store_true", default=False,
                        help="Demonstration mode (run a single eco).")

    parser.add_argument('-f', '--force', action="store_true", default=False,
                        help="Force creation of new data.")

    parser.add_argument('-n', '--n_good', action="store", default=3, type=int,
                        help="How many goods do you want (default = 3).")

    parsed_args = parser.parse_args()

    main(parsed_args)
