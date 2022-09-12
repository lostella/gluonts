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

from dataclasses import dataclass, field
import logging
from functools import singledispatch
from typing import Callable, Iterator, List, Optional, Any, Union, Type

import numpy as np

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import DataLoader
from gluonts.model.forecast import (
    Forecast,
    QuantileForecast,
    SampleForecast,
    Quantile,
)

logger = logging.getLogger(__name__)

OutputTransform = Callable[[DataEntry, np.ndarray], np.ndarray]

LOG_CACHE = set()
OUTPUT_TRANSFORM_NOT_SUPPORTED_MSG = (
    "The `output_transform` argument is not supported and will be ignored."
)
NOT_SAMPLE_BASED_MSG = (
    "Forecast is not sample based. Ignoring parameter `num_samples` from"
    " predict method."
)


def log_once(msg):
    global LOG_CACHE
    if msg not in LOG_CACHE:
        logger.info(msg)
        LOG_CACHE.add(msg)


# different deep learning frameworks generate predictions and the tensor to
# numpy conversion differently, use a dispatching function to prevent needing
# a ForecastGenerators for each framework
@singledispatch
def predict_to_numpy(prediction_net, args) -> np.ndarray:
    raise NotImplementedError


@singledispatch
def to_numpy(object):
    raise NotImplementedError


@to_numpy.register(tuple)
def _(t: tuple):
    return tuple(to_numpy(el) for el in t)


@to_numpy.register(list)
def _(t: list):
    return [to_numpy(el) for el in t]


@singledispatch
def _unpack(batched) -> Iterator:
    """
    Unpack batches.

    This assumes that arrays are wrapped in a  nested structure of lists and
    tuples, and each array has the same shape::

        >>> a = np.arange(5)
        >>> batched = [a, (a, [a, a, a])]
        >>> list(_unpack(batched))
        [[0, (0, [0, 0, 0])],
         [1, (1, [1, 1, 1])],
         [2, (2, [2, 2, 2])],
         [3, (3, [3, 3, 3])],
         [4, (4, [4, 4, 4])]]
    """

    if isinstance(batched, (list, tuple)):
        T = type(batched)

        return map(T, zip(*map(_unpack, batched)))

    return batched


@singledispatch
def make_distribution_forecast(distr, *args, **kwargs) -> Forecast:
    raise NotImplementedError


class ForecastBatch:
    @property
    def batch_size(self) -> int:
        raise NotImplementedError

    @property
    def mean(self) -> np.ndarray:
        raise NotImplementedError

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class DistributionForecastBatch(ForecastBatch):
    start_date: list
    item_id: Optional[list]
    info: Optional[list]
    distr_output: Any  # TODO fix
    distr_args: list
    distr: Type = field(init=False)

    def __post_init__(self):
        self.distr = self.distr_output.distribution(*self.distr_args)

    def __iter__(self) -> Iterator[Forecast]:  # TODO fix
        distributions = [
            self.distr_output.distribution(*u)
            for u in _unpack(self.distr_args)
        ]

        for i, distr in enumerate(distributions):
            yield make_distribution_forecast(
                distr,
                start_date=self.start_date[i],
                item_id=self.item_id[i] if self.item_id is not None else None,
                info=self.info[i] if self.info is not None else None,
            )

    @property
    def mean(self) -> np.ndarray:
        return to_numpy(self.distr.mean())

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        return to_numpy(self.distr.quantile(q))


@dataclass
class SampleForecastBatch(ForecastBatch):
    start_date: list
    item_id: Optional[list]
    info: Optional[list]
    sample_batch: np.ndarray

    def __post_init__(self):
        self._sorted_samples_value = None

    @property
    def _sorted_samples(self) -> np.ndarray:
        if self._sorted_samples_value is None:
            self._sorted_samples_value = np.sort(self.sample_batch, axis=1)
        return self._sorted_samples_value

    def __iter__(self) -> Iterator[SampleForecast]:
        for i, sample in enumerate(self.sample_batch):
            yield SampleForecast(
                sample,
                start_date=self.start_date[i],
                item_id=self.item_id[i] if self.item_id is not None else None,
                info=self.info[i] if self.info is not None else None,
            )

    @property
    def batch_size(self) -> int:
        return self.sample_batch.shape[0]

    @property
    def num_samples(self) -> int:
        return self.sample_batch.shape[1]

    @property
    def mean(self) -> np.ndarray:
        if self._mean is not None:
            return self._mean
        else:
            return np.mean(self.samples, axis=1)

    def quantile(self, q: Union[float, str]) -> np.ndarray:
        q = Quantile.parse(q).value
        sample_idx = int(np.round((self.num_samples - 1) * q))
        return self._sorted_samples[:, sample_idx, :]


@dataclass
class QuantileForecastBatch(ForecastBatch):
    start_date: list
    item_id: Optional[list]
    info: Optional[list]
    quantile_batch: np.ndarray
    quantile_levels: List[str]

    def __iter__(self) -> Iterator[QuantileForecast]:
        for i, output in enumerate(self.quantile_batch):
            yield QuantileForecast(
                output,
                start_date=self.start_date[i],
                item_id=self.item_id[i] if self.item_id is not None else None,
                info=self.info[i] if self.info is not None else None,
                forecast_keys=self.quantile_levels,
            )
