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


def get_parameters():

    # ----- set these params ------ #
    x0 = 50
    range_repartitions = range(10, 40, 5)
    step = 2
    alpha_range = np.linspace(0.1, 0.5, step)
    beta_range = np.linspace(0.75, 1.5, step)
    gamma_range = np.linspace(0.001, 0.15, step)
    t_max = 100
    economy_model = "prod: i-1"
    agent_model = RLOnAcceptanceAgent

    n_good = 3

    # ------------------------------ #

    repartition = list(itertools.product(range_repartitions, repeat=n_good))

    # ---------- #

    var_param = itertools.product(alpha_range, beta_range, gamma_range, repartition)

    # ----------- #

    parameters = []

    for alpha, beta, gamma, rpt in var_param:
        param = {
            "cognitive_parameters": (alpha, beta, gamma),
            "repartition": rpt,  # (x0, ) + rpt,
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


def _produce_data():

    tqdm.write("Run simulations.")

    param = get_parameters()

    max_ = len(param)

    data = backup.structure.Data(n=len(param))

    with multiprocessing.Pool(processes=os.cpu_count()) as p:

        with tqdm(total=max_) as pbar:
            for pr, b in p.imap_unordered(_run, param):
                data.append(backup=b, param=pr)
                pbar.update()

    return data


def _demo_mode():

    parameters = {
        "repartition": [30, 30, 60],
        "economy_model": "prod: i-1",
        "agent_model": RLOnAcceptanceAgent,
        "cognitive_parameters": (0.1, 1, 0.1),
        "t_max": 200,
    }

    e = Economy(**parameters)

    bkp = e.run()
    bkp.update(parameters)
    graph.single(bkp)


def _running_mode(args):

    if args.force or not os.path.exists("data/phase.p"):
        bkp = _produce_data()
        backup.backup.save(obj=bkp, file="phase.p")

    else:
        bkp = backup.backup.load("phase.p")

    graph.run(bkp)


def main(args):

    if args.demo:
        _demo_mode()

    else:
        _running_mode(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run money simulations.')

    parser.add_argument('-d', '--demo', action="store_true", default=False,
                        help="Demonstration mode (run a single eco).")

    parser.add_argument('-f', '--force', action="store_true", default=False,
                        help="Force creation of new data.")

    parsed_args = parser.parse_args()

    main(parsed_args)
