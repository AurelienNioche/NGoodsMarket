from model.economy import Economy

from model.rl_agent import RLOnAcceptanceAgent
from analysis import graph
# from model.stupid_agent import StupidAgent
# from model.frequentist_agent import FrequentistAgent


def main():

    cognitive_parameters = {
        "alpha": 0.1,
        "temp": 0.01,
        # "memory_span": 10,
        "u": 1
    }

    parameters = {
        "repartition_of_roles": [30, 30, 60],
        "economy_model": "prod: i-1",
        "agent_model": RLOnAcceptanceAgent,
        "cognitive_parameters": cognitive_parameters,
        "t_max": 200
    }
    e = Economy(
        **parameters
    )

    backup = e.run()
    graph.represent_results(backup, parameters)


if __name__ == "__main__":

    main()
