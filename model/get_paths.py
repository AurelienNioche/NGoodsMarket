import itertools as it


def get_paths(final_node, n_nodes):
    all_nodes_except_final = [i for i in range(n_nodes) if i != final_node]

    all_paths = dict()

    for departure_node in all_nodes_except_final:

        step_nodes = [i for i in range(n_nodes) if i not in [final_node, departure_node]]

        paths = [[(departure_node, final_node)]]

        for i in range(1, len(step_nodes) + 1):

            for j in it.permutations(step_nodes, r=i):
                node_list = [departure_node] + list(j) + [final_node]
                path = [(node_list[i], node_list[i + 1]) for i in range(len(node_list) - 1)]
                paths.append(path)

        all_paths[departure_node] = paths

    return all_paths


if __name__ == "__main__":

    print(get_paths(final_node=2, n_nodes=4))
