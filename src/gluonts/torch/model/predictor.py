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

from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn

from gluonts.core.serde import dump_json, load_json
from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.model.forecast import Forecast
from gluonts.model.forecast_generator import (
    ForecastBatch,
    predict_to_numpy,
    to_numpy,
)
from gluonts.model.predictor import Predictor
from gluonts.torch.batchify import batchify
from gluonts.torch.component import equals
from gluonts.transform import Transformation


@predict_to_numpy.register(nn.Module)
def _(prediction_net: nn.Module, args) -> np.ndarray:
    return prediction_net(*args).cpu().numpy()


@to_numpy.register(torch.Tensor)
def _(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().numpy()


class PyTorchPredictor(Predictor):
    def __init__(
        self,
        input_names: List[str],
        prediction_net: nn.Module,
        batch_size: int,
        prediction_length: int,
        input_transform: Transformation,
        lead_time: int = 0,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__(prediction_length, lead_time=lead_time)
        self.input_names = input_names
        self.prediction_net = prediction_net.to(device)
        self.batch_size = batch_size
        self.input_transform = input_transform
        self.device = device

    def to(self, device) -> "PyTorchPredictor":
        self.prediction_net = self.prediction_net.to(device)
        self.device = device
        return self

    @property
    def network(self) -> nn.Module:
        return self.prediction_net

    def predict(self, dataset: Dataset) -> Iterator[Forecast]:
        for forecast_batch in self.predict_batches(dataset):
            yield from forecast_batch

    def predict_batches(self, dataset: Dataset) -> Iterator[ForecastBatch]:
        inference_data_loader = InferenceDataLoader(
            dataset,
            transform=self.input_transform,
            batch_size=self.batch_size,
            stack_fn=lambda data: batchify(data, self.device),
        )

        self.prediction_net.eval()

        with torch.no_grad():
            for batch in inference_data_loader:
                yield self.prediction_net.forecast(batch)

    def __eq__(self, that):
        if type(self) != type(that):
            return False

        if not equals(self.input_transform, that.input_transform):
            return False

        return equals(
            self.prediction_net.state_dict(),
            that.prediction_net.state_dict(),
        )

    def serialize(self, path: Path) -> None:
        super().serialize(path)

        # serialize network
        with (path / "prediction_net.json").open("w") as fp:
            print(dump_json(self.prediction_net), file=fp)
        torch.save(
            self.prediction_net.state_dict(), path / "prediction_net_state"
        )

        # serialize transformation chain
        with (path / "input_transform.json").open("w") as fp:
            print(dump_json(self.input_transform), file=fp)

        # serialize all remaining constructor parameters
        with (path / "parameters.json").open("w") as fp:
            parameters = dict(
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                lead_time=self.lead_time,
                input_names=self.input_names,
            )
            print(dump_json(parameters), file=fp)

    @classmethod
    def deserialize(
        cls, path: Path, device: Optional[torch.device] = None
    ) -> "PyTorchPredictor":
        # deserialize constructor parameters
        with (path / "parameters.json").open("r") as fp:
            parameters = load_json(fp.read())

        # deserialize transformation chain
        with (path / "input_transform.json").open("r") as fp:
            transformation = load_json(fp.read())

        # deserialize network
        with (path / "prediction_net.json").open("r") as fp:
            prediction_net = load_json(fp.read())
        prediction_net.load_state_dict(
            torch.load(path / "prediction_net_state", map_location=device)
        )

        parameters["device"] = device

        return PyTorchPredictor(
            input_transform=transformation,
            prediction_net=prediction_net,
            **parameters,
        )
