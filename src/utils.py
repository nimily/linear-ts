import numpy as np
import numpy.linalg as npl
import numpy.random as npr


class DataSummary:
    prior_var: float

    xy: np.ndarray
    xx: np.ndarray

    _mean: np.ndarray
    _basis: np.ndarray
    _scale: np.ndarray
    _dirty: bool

    def __init__(self, dim, prior_var):
        self.prior_var = prior_var
        self.param_bound = (dim * self.prior_var) ** 0.5  # FIXME

        self.xy = np.zeros(dim, dtype=np.float)
        self.xx = np.eye(dim, dtype=np.float) / prior_var

        self._mean = np.zeros(dim, dtype=np.float)
        self._basis = np.eye(dim, dtype=np.float)
        self._scale = np.ones(dim, dtype=np.float) / prior_var
        self._dirty = False

    def _update_caches(self):
        svd = npl.svd(self.xx, hermitian=True)

        self._mean = svd[0] @ ((svd[2] @ self.xy) / svd[1])
        self._basis = svd[2]
        self._scale = svd[1]
        self._dirty = False

    def add_obs(self, x, y, tau=1.0):
        self.xy += x * y / tau ** 2
        self.xx += np.outer(x, x) / tau ** 2

        self._dirty = True

    @property
    def d(self):
        return self.xy.shape[0]

    @property
    def lambda_(self):
        return 1.0 / self.prior_var

    @property
    def mean(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._mean

    @property
    def basis(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._basis

    @property
    def scale(self) -> np.ndarray:
        if self._dirty:
            self._update_caches()

        return self._scale

    @property
    def thinness(self):
        scale_inv = 1 / self.scale
        return (max(scale_inv) * self.d / sum(scale_inv)) ** 0.5

    def radius_det(self, delta=1e-4):
        term1 = np.log(self.scale / self.lambda_).sum() - 2 * np.log(delta)
        term2 = self.lambda_ * self.param_bound ** 2

        return term1 ** 0.5 + term2 ** 0.5


class MetricAggregator:
    def __init__(self):
        self.m0 = []
        self.m1 = []
        self.m2 = []

    def confidence_band(self):
        m0 = np.array(self.m0)
        m1 = np.array(self.m1)
        m2 = np.array(self.m2)

        m0 = np.maximum(m0, 1)

        mean = m1 / m0
        var = (m2 - m1 ** 2 / m0) / (m0 - 1)
        sd = var ** 0.5
        se = (var / m0) ** 0.5

        return mean, sd, se

    def plot(self, ax, label, scale=2.0):
        mean, sd, se = self.confidence_band()

        x = np.arange(len(mean))

        lower = mean - scale * se
        upper = mean + scale * se

        ax.fill_between(x, lower, upper, alpha=0.2)
        ax.plot(x, mean, label=label)

    def aggregate(self, xs, filter=lambda _: True):
        self._ensure_len(len(xs))

        for i, x in enumerate(xs):
            if filter(i):
                self.m0[i] += 1
                self.m1[i] += x
                self.m2[i] += x ** 2

    def _ensure_len(self, n):
        dn = n - len(self.m0)

        if dn > 0:
            self.m0 += [0] * dn
            self.m1 += [0] * dn
            self.m2 += [0] * dn


class StateFactory:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self):
        state = npr.RandomState(self.seed)

        self.seed += 1

        return state
