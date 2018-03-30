import numpy as np


def array_summing_to_s(s, n):

    x = sorted(np.random.choice(np.arange(1, s+n), size=n-1, replace=False))

    out = []
    for i in range(n):
        if i == n-1:
            out.append(
                s+(n-1) - x[n-2]
            )
        elif i == 0:
            out.append(
                x[0] - 1
            )
        else:

            out.append(
                x[i] - x[i-1] - 1
            )

    return out


def softmax(x, temp):
    return np.exp(x / temp) / np.sum(np.exp(x / temp))


def main():

    print(array_summing_to_s(s=10, n=3))

if __name__ == "__main__":

    main()