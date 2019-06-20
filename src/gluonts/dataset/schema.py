from typing import List, Dict, Any, Optional

import numpy as np

from gluonts.dataset.common import DataEntry, Dataset
from gluonts.transform import FieldName


class DatasetSchema:

    mandatory_fields: List[str] = [FieldName.START, FieldName.TARGET]

    optional_fields: List[str] = [
        FieldName.FEAT_DYNAMIC_CAT,
        FieldName.FEAT_DYNAMIC_REAL,
        FieldName.FEAT_STATIC_CAT,
        FieldName.FEAT_STATIC_REAL,
    ]

    schema: Optional[Dict[str, Any]] = None

    def __init__(self, dataset: Dataset) -> None:
        for entry in dataset:
            if self.schema is None:
                self._init_schema(entry)
            else:
                self._merge_entry(entry)

    @classmethod
    def _get_standard_field_names(cls, data_entry: DataEntry) -> List[str]:
        assert all(k in data_entry.keys() for k in cls.mandatory_fields)
        return cls.mandatory_fields + [
            k for k in cls.optional_fields if k in data_entry.keys()
        ]

    @staticmethod
    def _target_dimension(target: np.ndarray) -> int:
        return target.shape[0] if len(target.shape) > 1 else 0

    def _init_schema(self, entry: DataEntry) -> None:
        assert self.schema is None

        field_names = self._get_standard_field_names(entry)

        self.schema = {
            FieldName.START: {},
            FieldName.TARGET: {"shape": entry[FieldName.TARGET].shape[:-1]},
        }

        if FieldName.FEAT_DYNAMIC_CAT in field_names:
            self.schema[FieldName.FEAT_DYNAMIC_CAT] = {
                "shape": entry[FieldName.FEAT_DYNAMIC_CAT].shape[:-1],
                "cardinality": (
                    entry[FieldName.FEAT_DYNAMIC_CAT].max(axis=0) + 1
                ),
            }

        if FieldName.FEAT_DYNAMIC_REAL in field_names:
            self.schema[FieldName.FEAT_DYNAMIC_REAL] = {
                "shape": entry[FieldName.FEAT_DYNAMIC_REAL].shape[:-1]
            }

        if FieldName.FEAT_STATIC_CAT in field_names:
            self.schema[FieldName.FEAT_STATIC_CAT] = {
                "shape": entry[FieldName.FEAT_STATIC_CAT].shape,
                "cardinality": entry[FieldName.FEAT_STATIC_CAT] + 1,
            }

        if FieldName.FEAT_STATIC_REAL in field_names:
            self.schema[FieldName.FEAT_STATIC_REAL] = {
                "shape": entry[FieldName.FEAT_STATIC_REAL].shape
            }

    def _merge_entry(self, entry: DataEntry) -> None:
        assert self.schema is not None

        field_names = self._get_standard_field_names(entry)

        assert all(
            k in field_names for k in self.schema.keys()
        ), "an entry is missing some fields"
        assert all(
            k in self.schema.keys() for k in field_names
        ), "an entry has fields in excess"

        assert (
            entry[FieldName.TARGET].shape[:-1]
            == self.schema[FieldName.TARGET]["shape"]
        )

        if FieldName.FEAT_DYNAMIC_CAT in field_names:
            assert (
                entry[FieldName.FEAT_DYNAMIC_CAT].shape[:-1]
                == self.schema[FieldName.FEAT_DYNAMIC_CAT]["shape"]
            )
            self.schema[FieldName.FEAT_DYNAMIC_CAT][
                "cardinality"
            ] = np.maximum(
                self.schema[FieldName.FEAT_DYNAMIC_CAT]["cardinality"],
                entry[FieldName.FEAT_DYNAMIC_CAT].max(axis=0) + 1,
            )

        if FieldName.FEAT_DYNAMIC_REAL in field_names:
            assert (
                entry[FieldName.FEAT_DYNAMIC_REAL].shape[:-1]
                == self.schema[FieldName.FEAT_DYNAMIC_REAL]["shape"]
            )

        if FieldName.FEAT_STATIC_CAT in field_names:
            assert (
                entry[FieldName.FEAT_STATIC_CAT].shape
                == self.schema[FieldName.FEAT_STATIC_CAT]["shape"]
            )
            self.schema[FieldName.FEAT_STATIC_CAT]["cardinality"] = np.maximum(
                self.schema[FieldName.FEAT_STATIC_CAT]["cardinality"],
                entry[FieldName.FEAT_STATIC_CAT] + 1,
            )

        if FieldName.FEAT_STATIC_REAL in field_names:
            assert (
                entry[FieldName.FEAT_DYNAMIC_REAL].shape
                == self.schema[FieldName.FEAT_DYNAMIC_REAL]["shape"]
            )

    def _validate_entry(self, entry: DataEntry):
        assert self.schema is not None

        field_names = self._get_standard_field_names(entry)

        assert all(
            k in field_names for k in self.schema.keys()
        ), "an entry is missing some fields"
        assert all(
            k in self.schema.keys() for k in field_names
        ), "an entry has fields in excess"

        assert (
            entry[FieldName.TARGET].shape[:-1]
            == self.schema[FieldName.TARGET]["shape"]
        )

        if FieldName.FEAT_DYNAMIC_CAT in field_names:
            assert (
                entry[FieldName.FEAT_DYNAMIC_CAT].shape[:-1]
                == self.schema[FieldName.FEAT_DYNAMIC_CAT]["shape"]
            )
            assert np.all(
                entry[FieldName.FEAT_DYNAMIC_CAT].max(axis=0)
                < self.schema[FieldName.FEAT_DYNAMIC_CAT]["cardinality"]
            )

        if FieldName.FEAT_DYNAMIC_REAL in field_names:
            assert (
                entry[FieldName.FEAT_DYNAMIC_REAL].shape[:-1]
                == self.schema[FieldName.FEAT_DYNAMIC_REAL]["shape"]
            )

        if FieldName.FEAT_STATIC_CAT in field_names:
            assert (
                entry[FieldName.FEAT_STATIC_CAT].shape
                == self.schema[FieldName.FEAT_STATIC_CAT]["shape"]
            )
            assert np.all(
                entry[FieldName.FEAT_STATIC_CAT]
                < self.schema[FieldName.FEAT_STATIC_CAT]["cardinality"]
            )

        if FieldName.FEAT_STATIC_REAL in field_names:
            assert (
                entry[FieldName.FEAT_DYNAMIC_REAL].shape
                == self.schema[FieldName.FEAT_DYNAMIC_REAL]["shape"]
            )

    def validate(self, dataset: Dataset) -> None:
        for entry in dataset:
            self._validate_entry(entry)
