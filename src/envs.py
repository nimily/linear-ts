import abc

from typing import Optional

import numpy as np
import numpy.linalg as npl
import numpy.random as npr


class Environment:
    t: int
    param: np.ndarray
    ctx: Optional["Context"]
    ctx_gen: "ContextGenerator"
    noise_gen: "NoiseGenerator"

    def __init__(
        self,
        param: np.ndarray,
        ctx_gen: "ContextGenerator",
        noise_gen: "NoiseGenerator",
    ):
        self.t = -1
        self.param = param
        self.ctx = None
        self.ctx_gen = ctx_gen
        self.noise_gen = noise_gen

    def next(self):
        self.generate_context()
        self.prepare_for_context()

        return self.ctx

    def generate_context(self):
        self.t += 1
        self.ctx = self.ctx_gen.generate(self.t)

    def prepare_for_context(self):
        pass

    def get_feedback(self, arm_idx, ctx=None):
        if ctx is None:
            ctx = self.ctx

        mean = ctx.arms @ self.param
        noise = self.noise_gen.generate(ctx, arm_idx)
        max_rew = mean.max()

        return Feedback(ctx, arm_idx, mean[arm_idx], noise, max_rew)


class Context:
    def __init__(self, t, arms):
        self.t = t
        self.arms = arms


class Feedback:
    ctx: Context
    arm_idx: int

    noise: float
    max_rew: float
    mean_rew: float

    def __init__(self, ctx, arm_idx, mean_rew, noise, max_rew):
        self.ctx = ctx
        self.arm_idx = arm_idx

        self.noise = noise
        self.max_rew = max_rew
        self.mean_rew = mean_rew

    @property
    def t(self):
        return self.ctx.t

    @property
    def arms(self):
        return self.ctx.arms

    @property
    def chosen_arm(self):
        return self.ctx.arms[self.arm_idx]

    @property
    def rew(self):
        return self.mean_rew + self.noise

    @property
    def regret(self):
        return self.max_rew - self.mean_rew

    def __repr__(self):
        return f"LinFb(arm={self.arm_idx}, reg={self.regret}, noise={self.noise}, mean={self.mean_rew})"


class NoiseGenerator:
    def __init__(self, state=npr):
        self.state = state

    @abc.abstractmethod
    def generate(self, ctx, arm_idx):
        pass

    @staticmethod
    def gaussian_noise(sd: float, state=npr):
        class GaussianGenerator(NoiseGenerator):
            def generate(self, ctx, arm_idx):
                return self.state.randn() * sd

        return GaussianGenerator(state)


class ContextGenerator:
    @abc.abstractmethod
    def generate(self, t) -> Context:
        pass


class StochasticContextGenerator(ContextGenerator):
    def __init__(self, k, d, bound, state=npr):
        self.k = k
        self.d = d
        self.bound = bound

        self.state = state

    def generate(self, t):
        arms = self.state.uniform(-self.bound, self.bound, (self.k, self.d))

        return Context(t, arms)
