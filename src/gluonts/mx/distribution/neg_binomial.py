# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Dict, List, Optional, Tuple

import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .deterministic import DeterministicOutput
from .distribution import Distribution, _sample_multiple, getF, softplus
from .distribution_output import DistributionOutput
from .mixture import MixtureDistributionOutput


class NegativeBinomial(Distribution):
    r"""
    Negative binomial distribution, i.e. the distribution of the number of
    successes in a sequence of independent Bernoulli trials, before a given
    number of failures is observed.

    Parameters
    ----------
    k
        Tensor containing the number of failures to stop the Bernoulli trials.
    logit
        Tensor containing the log-odds of success for the Bernoulli trials.
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, k: Tensor, logit: Tensor) -> None:
        self.k = k
        self.logit = logit

    @property
    def F(self):
        return getF(self.k)

    @property
    def batch_shape(self) -> Tuple:
        return self.k.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        F = self.F
        log_binomial_coeff = (
            F.gammaln(x + self.k) - F.gammaln(1.0 + x) - F.gammaln(self.k)
        )
        p = F.sigmoid(self.logit)
        log_unnormalized_prob = x * F.log(p) + self.k * F.log1p(-p)
        return log_binomial_coeff + log_unnormalized_prob

    @property
    def mean(self) -> Tensor:
        return self.k * self.F.exp(self.logit)

    @property
    def stddev(self) -> Tensor:
        p = self.F.sigmoid(self.logit)
        return self.F.sqrt(self.mean / (1.0 - p))

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.float32
    ) -> Tensor:
        def s(k: Tensor, logit: Tensor) -> Tensor:
            F = self.F
            rate = F.random.gamma(alpha=k, beta=F.exp(logit))
            return F.random.poisson(lam=rate, dtype=dtype)

        return _sample_multiple(
            s, k=self.k, logit=self.logit, num_samples=num_samples
        )

    @property
    def args(self) -> List:
        return [self.k, self.logit]


class NegativeBinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"k": 1, "logit": 1}
    distr_cls: type = NegativeBinomial

    @classmethod
    def domain_map(cls, F, k, logit):
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon
        k = F.maximum(softplus(F, k), epsilon)
        logit = F.clip(logit, a_min=-10, a_max=10)
        return k.squeeze(axis=-1), logit.squeeze(axis=-1)

    # Overwrites the parent class method.
    # We cannot scale using the affine transformation since negative binomial should return integers.
    # Instead we scale the parameters.
    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> NegativeBinomial:
        k, logit = distr_args
        if scale is None:
            return NegativeBinomial(k, logit)
        else:
            F = getF(k)
            k = F.broadcast_mul(k, scale)
            return NegativeBinomial(k, logit, F)

    @property
    def event_shape(self) -> Tuple:
        return ()


class ZeroInflatedNegativeBinomialOutput(MixtureDistributionOutput):
    def __init__(self):
        super().__init__([NegativeBinomialOutput(), DeterministicOutput(0)])
