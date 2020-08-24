import argparse

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

from utils import StateFactory


def argmax(array):
    rows = range(array.shape[0])
    cols = array.argmax(axis=1)

    return rows, cols


def compute_prob(n_block, sigma, tau, state_factory):
    state = state_factory()

    def gaussian():
        return state.randn(n_block, 2)

    theta = gaussian() * sigma

    # making one observation per entry
    mean = (theta + gaussian() * tau) / 2

    # choosing between e0 and e1 within each block
    sample = mean + gaussian() / 2 ** 0.5
    index = argmax(sample)

    # updating the mean given new rewards
    rew = theta + gaussian() * tau
    mean[index] = (2 * mean + rew)[index] / 3

    # computing the directional mean and variance
    dir_mean = mean.sum()
    dir_var = n_block * (1 / 2 + 1 / 3)

    ratio = (tau - sigma) * dir_mean / dir_var ** 0.5

    return theta.sum() * (tau - sigma) > 0, stats.norm.cdf(ratio)


def run_experiments(n_iter, sigma, tau, seed):
    state_factory = StateFactory(seed)

    n_blocks = 2 ** np.arange(18)
    n_failure = np.zeros_like(n_blocks, dtype=np.float64)

    for i, n_block in enumerate(n_blocks):
        results = [
            compute_prob(n_block, sigma, tau, state_factory) for _ in range(n_iter)
        ]

        max_prob = max(p for _, p in results)
        min_fail = int(1 / max_prob)

        n_subopt = sum(i for i, _ in results) / len(results)

        print(
            f"Number of blocks = {n_block:7d} -- proportion of times 0 is suboptimal = {n_subopt:.2f} -- "
            f"min number of steps TS will fail = {min_fail}"
        )

        n_failure[i] = min_fail

    # plotting the data
    plt.plot(n_blocks, np.log(n_failure))
    plt.scatter(n_blocks, np.log(n_failure))

    plt.xlabel("number of blocks (half of the dimension)")
    plt.ylabel("log(expectation of failure time)")

    plt.savefig("plots/example-1.pdf")


def __main__():
    parser = argparse.ArgumentParser(
        description="Simulate the first example for TS failure."
    )

    parser.add_argument("--n-iter", type=int, help="number of iterations", default=50)
    parser.add_argument("--sigma", type=float, help="prior sd", default=1.0)
    parser.add_argument("--tau", type=float, help="noise sd", default=0.0)
    parser.add_argument("--seed", type=int, help="initial random seed", default=1)

    args = parser.parse_args()

    run_experiments(n_iter=args.n_iter, sigma=args.sigma, tau=args.tau, seed=args.seed)


if __name__ == "__main__":
    __main__()
