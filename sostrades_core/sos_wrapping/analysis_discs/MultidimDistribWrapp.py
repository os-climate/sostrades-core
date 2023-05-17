from typing import Union
import chaospy as cp
import openturns as ot
import numpy as np
from scipy.stats import norm


class MultiDimensionalDistribution:
    """
    Metaclass concatenating a list of unidimensional distributions to generate samples that can be either
    floats (unidimensional case) or arrays (multidimensional case).
    The purpose of this class is to concatenate distributions, thus in the multidimensional case,
    the random variables are independant from one another.
     """
    def __init__(self, lower_bound: Union[float, np.ndarray], upper_bound: Union[float, np.ndarray]):
        self.dim: int = 1
        if isinstance(lower_bound, np.ndarray):
            self.dim = len(lower_bound)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.distribs: list[ot.Distribution]  = []
        self._set_distributions()

    def getSample(self, sample_size: int) -> list[Union[float, np.ndarray]]:
        """Generate samples"""
        if len(self.distribs) == 0:
            raise Exception("Distributions are not set")
        samples = None
        if self.dim == 1:
            samples = self.distribs[0].getSample(size=sample_size)
        else:
            a = 1

        return samples

    def _set_distributions(self):
        raise NotImplementedError("This method should be overwritten in each subclasse")


class MultiDimensionalPERTDistribution(MultiDimensionalDistribution):
    def __init__(self,
                 lower_bound: Union[float, np.ndarray],
                 upper_bound: Union[float, np.ndarray],
                 most_probable_value: Union[float, np.ndarray]):
        self.most_probable_value = most_probable_value
        super().__init__(lower_bound=lower_bound, upper_bound=upper_bound)

    def _get_distribution(self,
                          lower_bound: float,
                          most_probable_value: float,
                          upper_bound: float,) -> ot.Distribution:
        """Returns a one dimensional PERT distribution"""
        chaospy_dist = cp.PERT(lower=lower_bound, mode=most_probable_value, upper=upper_bound)
        return ot.Distribution(ot.ChaospyDistribution(chaospy_dist))

    def _set_distributions(self):
        if self.dim == 1:
            self.distribs = [self._get_distribution(lower_bound=self.lower_bound,
                                                    upper_bound=self.upper_bound,
                                                    most_probable_value=self.most_probable_value)]
        else:
            self.distribs = [self._get_distribution(lb, mpv, ub) for lb, mpv, ub in zip(self.lower_bound,
                                                                                        self.most_probable_value,
                                                                                        self.upper_bound)]



class MultiDimensionalNormalDistribution(MultiDimensionalDistribution):
    def __init__(self,
                 lower_bound: Union[float, np.ndarray],
                 upper_bound: Union[float, np.ndarray],
                 confidence_interval: float):
        self.confidence_interval = confidence_interval
        super().__init__(lower_bound=lower_bound, upper_bound=upper_bound)


    def _get_distribution(self,
                          lower_bound: float,
                          upper_bound: float, ):
        """Returns a one dimensional PERT distribution"""
        norm_val = float(format(1 - self.confidence_interval, '.2f')) / 2
        ratio = norm.ppf(1 - norm_val) - norm.ppf(norm_val)

        mu = (lower_bound + upper_bound) / 2
        sigma = (upper_bound - lower_bound) / ratio
        distrib = ot.Normal(mu, sigma)
        return distrib

    def _set_distributions(self):
        if self.dim == 1:
            self.distribs = [self._get_distribution(lower_bound=self.lower_bound,
                                                    upper_bound=self.upper_bound)]
        else:
            self.distribs = [self._get_distribution(lb, mpv) for lb, mpv, ub in zip(self.lower_bound, self.upper_bound)]



class MultiDimensionalTriangularDistribution(MultiDimensionalPERTDistribution):
    def __init__(self,
                 lower_bound: Union[float, np.ndarray],
                 upper_bound: Union[float, np.ndarray],
                 most_probable_value: Union[float, np.ndarray]):
        super().__init__(lower_bound=lower_bound, upper_bound=upper_bound, most_probable_value=most_probable_value)


    def _get_distribution(self,
                          lower_bound: float,
                          most_probable_value: float,
                          upper_bound: float,):
        """Returns a one dimensional PERT distribution"""
        distrib = ot.Triangular(int(lower_bound), int(most_probable_value), int(upper_bound))
        return distrib


class MultiDimensionalLogNormalDistribution(MultiDimensionalNormalDistribution):
    def __init__(self,
                 lower_bound: Union[float, np.ndarray],
                 upper_bound: Union[float, np.ndarray],
                 confidence_interval: float):
        super().__init__(lower_bound=lower_bound, upper_bound=upper_bound, confidence_interval=confidence_interval)

    def _get_distribution(self,
                          lower_bound: float,
                          upper_bound: float, ):
        """Returns a one dimensional PERT distribution"""
        norm_val = float(format(1 - self.confidence_interval, '.2f')) / 2
        ratio = norm.ppf(1 - norm_val) - norm.ppf(norm_val)

        mu = (lower_bound + upper_bound) / 2
        sigma = (upper_bound - lower_bound) / ratio
        distrib = ot.LogNormal()
        distrib.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
        return distrib
