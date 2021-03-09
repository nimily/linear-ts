import argparse

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

from utils import StateFactory


def argmax(array):
    rows = range(array.shape[0])
    cols = array.argmax(axis=1)

    return rows, cols


def compute_prob(dim, mu, rho, sigma, tau, state_factory):
    state = state_factory()

    exclusive = state.randn(3 * dim) * (sigma ** 2 - rho) ** 0.5
    shared = mu + state.randn() * rho ** 0.5
    theta = shared + exclusive

    # making one observation per entry
    x0 = np.zeros(3 * dim)
    x0[:dim] = -1.0 / dim ** 0.5

    x1 = np.zeros(3 * dim)
    x1[:dim] = -1.0 / dim ** 0.5
    x1[dim:] = +1.0 / dim ** 0.5

    # check whether x1 is the optimal arm
    x1_optimal = theta @ x1 > theta @ x0 and theta @ x1 > 0

    # check whether x0 is the first choice
    sample = state.randn(3 * dim)
    x0_selected = sample @ x0 > 0 and sample @ x0 > sample @ x1

    # the observation after the first round
    y = theta @ x0 + state.randn() * tau
    mean = (y * x0) / (1 + x0 @ x0)

    # computing the probability that x0 is pulled at t=2
    mean0 = mean @ x0
    var0 = (x0 @ x0) / (1 + x0 @ x0)
    ratio0 = mean0 / var0 ** 0.5
    prob0 = stats.norm.cdf(ratio0)

    # computing the probability that x1 is pulled at t=2
    mean1 = mean @ x1
    var1 = x1 @ (x1 - x0 * (x0 @ x0) / (1 + x0 @ x0))
    ratio1 = mean1 / var1 ** 0.5
    prob1 = stats.norm.cdf(ratio1)

    # applying the union bound
    prob = prob0 + prob1

    return np.array([x1_optimal, x0_selected, 1.0 / prob])


def run_experiment(n_iter, dim, mu, rho, sigma, tau, seed):
    state_factory = StateFactory(seed)

    results = np.stack(
        [compute_prob(dim, mu, rho, sigma, tau, state_factory) for _ in range(n_iter)]
    )

    if rho > 0.5:
        print(results)
        exit(0)

    return results


def __main__():
    types = ["mu", "rho", "dim"]
    labels = {
        "mu": "$\\mu$",
        "rho": "$\\rho$",
        "dim": "$d$",
    }

    parser = argparse.ArgumentParser(
        description="Verify the Thompson sampling failures empirically."
    )

    parser.add_argument("--n-iter", type=int, help="number of iterations", default=50)
    parser.add_argument(
        "--n-value", type=int, help="number of different values to try", default=21
    )
    parser.add_argument(
        "--dim", type=int, help="number of iterations", default=50 * 1000
    )
    parser.add_argument("--mu", type=float, help="prior mean", default=0.01)
    parser.add_argument("--rho", type=float, help="prior correlation", default=0.0)
    parser.add_argument("--sigma", type=float, help="prior sd", default=1.0)
    parser.add_argument("--tau", type=float, help="noise sd", default=1.0)
    parser.add_argument("--seed", type=int, help="initial random seed", default=1)
    parser.add_argument(
        "--change", type=str, help="varying parameter", choices=types, default="dim"
    )

    args = parser.parse_args()

    xticks = None
    xrots = "0"
    if args.change == "mu":
        mus = np.linspace(0.0, args.mu, args.n_value)
        xticks = [f"{i:.2f}" for i in mus]
        xrots = "45"
    else:
        mus = np.repeat(args.mu, args.n_value)

    if args.change == "rho":
        rhos = np.linspace(0.0, args.rho, args.n_value)
        xticks = [f"{i:.2f}" for i in rhos]
        xrots = "45"
    else:
        rhos = np.repeat(args.rho, args.n_value)

    if args.change == "dim":
        dims = 2 ** np.arange(1, args.n_value + 1)
        args.dim = dims[-1]
        xticks = [f"$2^{{{i}}}$" for i in range(args.n_value)]
    else:
        dims = np.repeat(args.dim, args.n_value).astype(np.int)

    results = []
    for mu, rho, dim in zip(mus, rhos, dims):
        print(f"Running experiment with mu = {mu:.3f} rho = {rho:.3f} dim = {dim}")
        results.append(
            run_experiment(
                n_iter=args.n_iter,
                dim=dim,
                mu=mu,
                rho=rho,
                sigma=args.sigma,
                tau=args.tau,
                seed=args.seed,
            )
        )
    results = np.array(results)

    # plotting the data
    y = results[:, :, 2].T
    plt.yscale('log')
    plt.boxplot(y, positions=range(args.n_value), showfliers=False)
    plt.xticks(range(args.n_value), xticks, rotation=xrots)

    plt.xlabel(labels[args.change])
    plt.ylabel("Expected number of failures for LinTS")
    plt.tight_layout()

    plt.savefig(
        f"plots/example-2-{args.change}-{args.dim}-{args.mu}-{args.rho}-{args.seed}.pdf"
    )

    print(f"Ratio of times x1 is optimal:")
    print(f"\t{results[:, :, 0].mean(axis=1)}")

    print(f"Ratio of times x0 is selected at t=0:")
    print(f"\t{results[:, :, 1].mean(axis=1)}")

    print("Experiment finished successfully.")


if __name__ == "__main__":
    __main__()
