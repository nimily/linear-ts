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
    ctx_gen = CtxGenerator(k, d, arm_bound, state=state_factory())
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
    thinnesses = defaultdict(MetricAggregator)
    for i in range(n):
        print(f"Running experiment {i}...")
        results = russo_scenario(
            d=d,
            k=k,
            t=t,
            prior_var=10.0,
            arm_bound=1 / 10 ** 0.5,
            state_factory=state_factory,
        )

        for name, (regret, thinness) in results.items():
            regrets[name].aggregate(np.cumsum(regret))
            thinnesses[name].aggregate(thinness)

    for name, thinness in thinnesses.items():
        thinness.plot(plt, name)

    plt.legend()
    plt.savefig(f"plots/thinness-{n}-{d}-{k}-{t}.pdf")
    plt.show()

    for name, regret in regrets.items():
        regret.plot(plt, name)

    plt.legend()
    plt.savefig(f"plots/regret-{n}-{d}-{k}-{t}.pdf")
    plt.show()

    # aggregates = defaultdict(MetricAggregator)
    # for i in range(n):
    #     print(f'Running experiment [{i}]...')
    #     metrics = run_single_experiment(d, k, t, g, l, state_factory)
    #     for name, metric in metrics.items():
    #         aggregates[name].aggregate(np.cumsum(metric.regrets))
    #
    # for name, aggregate in aggregates.items():
    #     mean, sd, se = aggregate.confidence_band()
    #
    #     lower = mean - 2 * se
    #     upper = mean + 2 * se
    #
    #     plt.fill_between(range(t), lower, upper, alpha=0.2)
    #     plt.plot(range(t), mean, label=name)
    #
    # plt.legend()
    # plt.savefig(f'plots/regret-{n}-{d}-{k}-{t}-{s}-{"grouped" if g else "ungrouped"}.pdf')
    # plt.show()
    #
    # print()
    # print('  policy   |   regret')
    # print('-' * 25)
    # for name, aggregate in aggregates.items():
    #     mean = aggregate.confidence_band()[0][-1]
    #     print(f'{name:10} | {mean:.2f}')
    # print()
    #
    # print(f'All the experiments finished successfully.')


def __main__():
    parser = argparse.ArgumentParser(
        description="Run simulations for various ROFUL algorithms."
    )

    parser.add_argument("-n", type=int, help="number of iterations", default=50)
    parser.add_argument("-k", type=int, help="number of actions", default=10)
    parser.add_argument("-d", type=int, help="dimension", default=120)
    parser.add_argument("-t", type=int, help="time horizon", default=10000)
    parser.add_argument("-s", type=int, help="random seed", default=1)

    args = parser.parse_args()

    run_experiments(n=args.n, d=args.d, k=args.k, t=args.t, s=args.s)


if __name__ == "__main__":
    __main__()