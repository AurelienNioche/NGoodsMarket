import numpy as np


factor_medium_difference = 2
threshold_time_duration = 0.5


def run_with_medium(medium, m):

    t_max = len(medium)

    n_good = len(medium[0])

    good_count = np.zeros(n_good, dtype=int)

    last_good_in_memory = None

    for t in range(t_max):

        d = np.array(medium[t])
        idx = np.flatnonzero(d == max(d))

        # If there is one max value
        if len(idx) == 1:

            cond0 = last_good_in_memory == idx

            other_good0, other_good1 = d[d != d[idx]]
            cond1 = d[idx] > other_good0 * factor_medium_difference
            cond2 = d[idx] > other_good1 * factor_medium_difference

            if cond0 and cond1 and cond2:
                good_count[idx] += 1
            else:
                good_count[idx] = 0

            last_good_in_memory = idx
        else:
            # reset all goods count if max() returns two values
            good_count[idx] = 0

    cond0 = max(good_count) > (threshold_time_duration * t_max)
    cond1 = np.argmax(good_count) == m

    return cond0 and cond1


def run_with_exchange(exchange, m):

    t_max = len(exchange)

    score = 0

    for t in range(t_max):
        for k, v in exchange[t].items():

            if m in k:
                score += v
            else:
                score -= v

    return score
