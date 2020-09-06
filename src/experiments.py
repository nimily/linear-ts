import argparse
from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt

from envs import Environment
from envs import NoiseGenerator
from envs import StochasticContextGenerator as CtxGenerator

from policies import Roful

from utils import MetricAggregator
from utils import StateFactory


def russo_scenario(
    state_factory, k=100, d=100, t=1000, sd=1.0, prior_var=10.0, arm_bound=0.1
):
    param = state_factory().randn(d) * prior_var ** 0.5
    ctx_gen = CtxGenerator.uniform_on_sphere(k, d, arm_bound, state=state_factory())
    noise_gen = NoiseGenerator.gaussian_noise(sd, state=state_factory())

    env = Environment(param, ctx_gen, noise_gen)
    algs = {
        "TS-1": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=1.0),
        "TS-2": Roful.ts(d, prior_var=prior_var, state=state_factory(), inflation=5.0),
        "TS-3": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.conditional_inflation(5.0, thin_thresh=2.0),
        ),
        "TS-4": Roful.ts(
            d,
            prior_var=prior_var,
            state=state_factory(),
            inflation=Roful.radius_inflation(),
        ),
    }

    for i in range(t):
        ctx = env.next()

        for alg in algs.values():
            idx = alg.choose_arm(ctx)
            fb = env.get_feedback(idx)
            alg.update(fb)

        if i % 100 == 0:
            print(i)

    return {name: (alg.metrics.regrets, alg.thinnesses) for name, alg in algs.items()}


def run_experiments(n, d, k, t, s):
    state_factory = StateFactory(s + 1)

    regrets = defaultdict(MetricAggregator)
    cumregrets = defaultdict(MetricAggregator)
    thinnesses = defaultdict(MetricAggregator)
    for i in range(n):
        print(f"Running experiment {i}...")
        results = russo_scenario(
            d=d, k=k, t=t, prior_var=10.0, arm_bound=1.0, state_factory=state_factory,
        )

        for name, (regret, thinness) in results.items():
            regrets[name].aggregate(regret)
            cumregrets[name].aggregate(np.cumsum(regret))
            thinnesses[name].aggregate(thinness)

    metrics = {
        "regret": regrets,
        "cumregret": cumregrets,
        "thinnesses": thinnesses,
    }
    for name, metric in metrics.items():
        plt.clf()
        for alg, agg in metric.items():
            agg.plot(plt, alg)

        plt.legend()
        plt.savefig(f"plots/{name}-{n}-{d}-{k}-{t}.pdf")


def __main__():
    parser = argparse.ArgumentParser(
        description="Run simulations for various ROFUL algorithms."
    )

    parser.add_argument("-n", type=int, help="number of iterations", default=50)
    parser.add_argument("-k", type=int, help="number of actions", default=10)
    parser.add_argument("-d", type=int, help="dimension", default=100)
    parser.add_argument("-t", type=int, help="time horizon", default=10000)
    parser.add_argument("-s", type=int, help="random seed", default=1)

    args = parser.parse_args()

    run_experiments(n=args.n, d=args.d, k=args.k, t=args.t, s=args.s)


if __name__ == "__main__":
    __main__()
