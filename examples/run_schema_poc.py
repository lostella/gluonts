from gluonts.dataset.common import ListDataset
from gluonts.dataset.schema import DatasetSchema
from gluonts.model.deepar import DeepAREstimator

invalid_dataset = ListDataset(
    data_iter=[
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0]],
        },
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    ],
    freq="5min",
)

valid_dataset = ListDataset(
    data_iter=[
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0]],
            "feat_static_cat": [3, 4, 5],
        },
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "feat_static_cat": [1, 1, 1],
        },
    ],
    freq="5min",
)

test_dataset_1 = ListDataset(
    data_iter=[
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0]],
            "feat_static_cat": [3, 4, 5],
        },
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "feat_static_cat": [4, 0, 0],  # INVALID
        },
    ],
    freq="5min",
)

test_dataset_2 = ListDataset(
    data_iter=[
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0]],
            "feat_static_cat": [3, 4, 5],
        },
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            # INVALID
            "feat_static_cat": [3, 4, 5],
        },
    ],
    freq="5min",
)

test_dataset_3 = ListDataset(
    data_iter=[
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0]],
            "feat_static_cat": [3, 4],  # INVALID
        },
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            # INVALID
            "feat_static_cat": [3, 4, 5],
        },
    ],
    freq="5min",
)

test_dataset_4 = ListDataset(
    data_iter=[
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0],
            "feat_dynamic_real": [1.0, 2.0, 3.0, 4.0],  # INVALID
            "feat_static_cat": [3, 4, 5],
        },
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "feat_static_cat": [0, 0, 0],
        },
    ],
    freq="5min",
)

test_dataset_5 = ListDataset(
    data_iter=[
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0]],  # INVALID
            "feat_static_cat": [3, 4, 5],
        },
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
            "feat_static_cat": [0, 0, 0],
        },
    ],
    freq="5min",
)

test_dataset_6 = ListDataset(
    data_iter=[
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0]],
            "feat_static_cat": [1, 1, 1],
        },
        {
            "start": "2019-01-01 00:00:00",
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_dynamic_real": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "feat_static_cat": [3, 3, 3],
        },
    ],
    freq="5min",
)

try:
    dataset_schema = DatasetSchema(invalid_dataset)
    print(dataset_schema)
except AssertionError as e:
    print("invalid dataset")

dataset_schema = DatasetSchema(valid_dataset)
print(dataset_schema.schema)

invalid_test_datasets = [
    test_dataset_1,
    test_dataset_2,
    test_dataset_3,
    test_dataset_4,
    test_dataset_5
]

for test_dataset in invalid_test_datasets:
    try:
        dataset_schema.validate(test_dataset)
        print("valid test dataset")
    except AssertionError as e:
        print("invalid test dataset")

dataset_schema.validate(test_dataset_6)
print("valid test dataset")

# estimator = DeepAREstimator('H', prediction_length=1)
#
# predictor = estimator.train(valid_dataset)
#
# forecasts = predictor.predict(test_dataset_6)
#
# for f in forecasts:
#     print(f)
