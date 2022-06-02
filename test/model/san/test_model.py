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

import pytest

from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import QuantileForecast
from gluonts.mx.model.predictor import GluonPredictor
from gluonts.mx.trainer import Trainer
from gluonts.model.san import SelfAttentionEstimator
from gluonts.model.san._network import (
    SelfAttentionPredictionNetwork,
)


@pytest.fixture()
def hyperparameters():
    return dict(
        ctx="cpu",
        epochs=1,
        learning_rate=1e-3,
        hidden_dim=16,
        variable_dim=4,
        use_feat_dynamic_real=False,
        use_feat_dynamic_cat=False,
        use_feat_static_real=False,
        use_feat_static_cat=False,
    )


@pytest.mark.parametrize("hybridize", [True, False])
def test_accuracy(accuracy_test, hyperparameters, hybridize):
    hyperparameters.update(num_batches_per_epoch=50, hybridize=hybridize)
    accuracy_test(SelfAttentionEstimator, hyperparameters, accuracy=0.4)


def test_repr(repr_test, hyperparameters):
    repr_test(SelfAttentionEstimator, hyperparameters)


def test_serialize(serialize_test, hyperparameters):
    serialize_test(SelfAttentionEstimator, hyperparameters)


def test_quantile_levels(hyperparameters):
    dataset = ListDataset(
        [{"start": "2020-01-01", "target": [10.0] * 50}], freq="D"
    )

    estimator = SelfAttentionEstimator(
        freq="D",
        prediction_length=2,
        use_feat_dynamic_real=False,
        use_feat_dynamic_cat=False,
        use_feat_static_real=False,
        use_feat_static_cat=False,
        trainer=Trainer(epochs=1),
    )
    predictor = estimator.train(training_data=dataset)

    forecast = next(iter(predictor.predict(dataset)))

    assert isinstance(forecast, QuantileForecast)
    assert isinstance(predictor, GluonPredictor)
    assert isinstance(predictor.prediction_net, SelfAttentionPredictionNetwork)
    assert all(
        float(k) == q
        for k, q in zip(
            forecast.forecast_keys, predictor.prediction_net.output.quantiles
        )
    )
