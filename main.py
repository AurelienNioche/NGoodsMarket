import argparse
import itertools
from model.economy import Economy
from tqdm import tqdm
import numpy as np

from model.rl_agent import RLOnAcceptanceAgent
from analysis import graph
from backup import backup
import multiprocessing
import os


def get_parameters():

    # ----- set these params ------ #
    decision_rule = "epsilon" #softmax
    x0 = 20
    fixed_good = 0
    range_repartitions = range(10, 40, 5)
    repartitions = list(itertools.product(range_repartitions, repeat=2))
    seeds = range(len(repartitions))
    step = 3
    # ------------------------------ #

    alpha_range = np.linspace(0.1, 0.5, step)
    beta_range = np.linspace(0, 1, step)

    if decision_rule == "softmax":
        decision_param_range = np.linspace(0.01, 0.05, step)
    else:
        decision_param_range = np.linspace(0.1, 0.5, step)

    possible_cognitive_parameters = \
        list(itertools.product(alpha_range, beta_range, decision_param_range))

    bkup = {
        "meta_parameters":
            {
                "fixed_good": fixed_good,
                "fixed_type": "x0",
                "fixed_good_n": x0,
                "decision_rule": decision_rule,
                "repartitions": repartitions,
                "possible_cognitive_parameters": possible_cognitive_parameters,
                "seeds": seeds
            }
    }

    return bkup


def run(*args):

    repartition = args[0][0]
    possible_cognitive_parameters = args[0][1]
    seed = args[0][2]

    bkup = {}

    with tqdm(total=len(possible_cognitive_parameters)) as pbareco:

        for alpha, beta, decision_rule_param in possible_cognitive_parameters:

            cognitive_parameters = {
                "alpha": alpha,
                "beta": beta

            }

            if decision_rule_param == "softmax":
                cognitive_parameters["temp"] = decision_rule_param
            else:
                cognitive_parameters["epsilon"] = decision_rule_param

            param = {
                "repartition_of_roles": repartition,
                "economy_model": "prod: i-1",
                "agent_model": RLOnAcceptanceAgent,
                "cognitive_parameters": cognitive_parameters,
                "t_max": 200,
                "seed": seed
            }

            e = Economy(
                **param
            )

            bkup[(alpha, beta, decision_rule_param)] = e.run()
            bkup[(alpha, beta, decision_rule_param)]["parameters"] = param

            pbareco.update()

    return bkup


def main(args):

    cognitive_parameters = {
        "alpha": 0.1,
        "temp": 0.01,
        "beta": 1,
        # "memory_span": 10,
        "u": 1
    }

    if not args.phase:

        parameters = {
            "repartition_of_roles": [30, 30, 60],
            "economy_model": "prod: i-1",
            "agent_model": RLOnAcceptanceAgent,
            "cognitive_parameters": cognitive_parameters,
            "t_max": 200,
        }

        e = Economy(
            **parameters
        )

        bkup = e.run()
        graph.represent_results(bkup, parameters)

    else:

        # ---------------------- Produce data ------------------ #

        if args.force:

            bkup = get_parameters()

            x0 = bkup["meta_parameters"]["fixed_good_n"]
            cognitive_parameters = bkup["meta_parameters"]["possible_cognitive_parameters"]
            repartitions = bkup["meta_parameters"]["repartitions"]
            seeds = bkup["meta_parameters"]["seeds"]
            complete_repartitions = [(x0, x1, x2) for x1, x2 in repartitions]

            if args.multiprocessing:

                # -------------------- Using multiprocessed simulations ------------- #

                tqdm.write("Run simulations using multiprocessing.")

                pool = multiprocessing.Pool(processes=os.cpu_count() - 1)
                
                prepared_args = zip(
                    complete_repartitions,
                    [cognitive_parameters for _ in range(len(complete_repartitions))],
                    seeds,
                )

                values = pool.map(run, prepared_args)

                for r, v in zip(repartitions, values):
                    bkup[r] = {}
                    for c, data in v.items():
                        bkup[r][c] = data

            else:

                # -------------------- Using sequentials simulations ------------- #

                tqdm.write("Run simulations sequentially.")

                for r in complete_repartitions:

                    c, data = run(
                        r,
                        cognitive_parameters
                    ).items()

                    bkup[r][c] = data

            backup.save(obj=bkup, file="phase.p")

        else:
            bkup = backup.load("phase.p")

        graph.phase_diagram(bkup)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run money simulations.')

    parser.add_argument('-p', '--phase', action="store_true", default=False,
                        help="test multiple repartitions then plot a phase diagram.")

    parser.add_argument('-f', '--force', action="store_true", default=False,
                        help="Force creation of new data.")

    parser.add_argument('-ml', '--multiprocessing', action="store_true", default=False,
                        help="Force creation of new data.")

    parsed_args = parser.parse_args()

    main(parsed_args)
