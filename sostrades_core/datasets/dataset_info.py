"""
Copyright 2024 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    """
    Stores the informations of a dataset
    """
    # Keys for parsing json
    CONNECTOR_ID_KEY = "connector_id"
    DATASET_ID_KEY = "dataset_id"

    # Id of the connector
    connector_id: str
    # Dataset id for this connector
    dataset_id: str

    @staticmethod
    def deserialize(input_dict: dict) -> DatasetInfo:
        """
        Method to deserialize
        expected
        {
            "connector_id": <connector_id>,
            "dataset_id": <dataset_id>,
        }
        :param input_dict: dict like input json object
        :type input_dict: dict
        """
        return DatasetInfo(
            connector_id=input_dict[DatasetInfo.CONNECTOR_ID_KEY],
            dataset_id=input_dict[DatasetInfo.DATASET_ID_KEY],
        )
