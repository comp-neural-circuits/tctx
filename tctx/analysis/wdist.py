"""
Code to explore the PDF and CDF of weight distributions.

We use truncated lognormals to define the distribution of excitatory connections.
We scale that by -8 for inhibitory connections.
We represent the inhibitory connections with a negative number as a convention to be consistent
with the network simulator (NEST), although technically conductances must be positive.
"""
import scipy.interpolate
import scipy.stats as st
import numpy as np


def _approx_pdf_from_cdf(cdf, vmin, vmax, n_samples=10**5):
    """numerically approximate the Probability Density Function from the cumulative"""
    x = np.linspace(vmin, vmax, n_samples)

    mid = .5 * (x[:-1] + x[1:])
    derivative = np.diff(cdf(x)) / np.diff(x)

    return scipy.interpolate.interp1d(mid, derivative, fill_value=0., bounds_error=False)


def _approx_inv_cdf_from_cdf(cdf, vmin, vmax, n_samples=10**5):
    """numerically approximate the inverse of a Cumulative Distribution Function"""
    x = np.linspace(vmin, vmax, n_samples)
    return scipy.interpolate.interp1d(cdf(x), x, fill_value=0., bounds_error=False)


class TruncatedLognormal:
    """
    Represents a truncated, and possibly scaled, lognormal distribution.
    """

    def __init__(self, loc, scale, shape, vmax, g=1):
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.vmax = vmax
        self.g = g

        self.base_lognorm = st.lognorm(
            loc=self.loc,
            scale=self.scale,
            s=self.shape)

        self.base_lognorm_cdf_vmax = self.base_lognorm.cdf(self.vmax)

        self._pdf = _approx_pdf_from_cdf(self.cdf, *self.vrange)
        self._icdf = _approx_inv_cdf_from_cdf(self.cdf, *self.vrange)

    @property
    def vrange(self) -> tuple:
        """truncated range of X"""
        vrange = 0, self.vmax * self.g

        if self.g < 0:
            vrange = vrange[1], vrange[0]

        return vrange

    def linspace(self, num=50):
        """generate samples linearly on the domain of X"""
        return np.linspace(*self.vrange, num=num)

    def cdf(self, weight):
        """Cumulative Distribution Function"""
        weight_norm = weight / self.g
        prob = np.minimum(self.base_lognorm.cdf(weight_norm) / self.base_lognorm_cdf_vmax, 1)
        if self.g < 0:
            prob = 1 - prob
        return prob

    def pdf(self, weight):
        """Probability Density Function"""
        return self._pdf(weight)

    def inv_cdf(self, prob):
        """
        Inverse of the Cumulative Distribution Function.
        Maps from probability to values.
        """
        return self._icdf(prob)

    def rev_cdf(self, prob):
        """
        Reversed Cumulative Distribution Function.
        Cumulative summation is done right-to-left.
        """
        return 1 - self.cdf(prob)

    def mean(self):
        """Estimated mean from the distribution"""
        x = self.linspace(1_000_000)
        p = self.pdf(x)
        p = p / np.sum(p)
        mean = np.sum(x * p)
        return mean

    def var(self):
        """Estimated var from the distribution"""
        mean = self.mean()
        x = self.linspace(1_000_000)
        p = self.pdf(x)
        p = p / np.sum(p)
        mean = np.sum(np.square(x - mean) * p)
        return mean

    def std(self):
        """Estimated std from the distribution"""
        return np.sqrt(self.var())

    def quantile(self, q):
        """Estimated quantile from the distribution"""
        assert 0 <= q <= 1
        return self.inv_cdf(q).item()

    def median(self):
        """Estimated median from the distribution"""
        return self.quantile(.5)

    def min(self):
        """Min value of the distribution"""
        return self.quantile(0)

    def max(self):
        """Max value of the distribution"""
        return self.quantile(1)


class ConnDist:
    """Combination of exc and inh weight distributions"""

    def __init__(self, e_weights_loc, e_weights_scale, e_weights_shape, e_weights_vmax, g):
        assert g < 0

        self.exc = TruncatedLognormal(
            e_weights_loc,
            e_weights_scale,
            e_weights_shape,
            e_weights_vmax
        )
        self.inh = TruncatedLognormal(
            e_weights_loc,
            e_weights_scale,
            e_weights_shape,
            e_weights_vmax,
            g=g,
        )

    @classmethod
    def from_batch(cls, batch):
        param_names = ['e_weights_loc', 'e_weights_scale', 'e_weights_vmax', 'e_weights_shape', 'g']
        weight_dist_params = batch.reg[param_names].drop_duplicates()
        assert len(weight_dist_params) == 1
        weight_dist_params = weight_dist_params.iloc[0]

        return cls(**weight_dist_params)
