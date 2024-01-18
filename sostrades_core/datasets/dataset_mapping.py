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

import json

from sostrades_core.datasets.dataset_info import DatasetInfo


@dataclass()
class DatasetsMapping:
    """
    Stores namespace/dataset mapping
    """
    # Keys for parsing json
    DATASETS_INFO_KEY = "datasets_infos"
    NAMESPACE_KEY = "namespace_datasets_mapping"

    # Dataset info [dataset_name : DatasetInfo]
    datasets_infos: dict[str:DatasetInfo]
    # Namespace mapping [namespace_name : List[DatasetInfo]]
    namespace_datasets_mapping: dict[str : list[DatasetInfo]]

    @staticmethod
    def deserialize(input_dict: dict) -> DatasetsMapping:
        """
        Method to deserialize
        expected example
        {
            "datasets_infos": {
                "Dataset1": {
                    "connector_id": <connector_id>,
                    "dataset_id": <dataset_id>,
                },
                "Dataset2": {
                    "connector_id": <connector_id>,
                    "dataset_id": <dataset_id>,
                }
            },
            "namespace_datasets_mapping": {
                "namespace1" : ["Dataset1"],
                "namespace2" : ["Dataset1", "Dataset2"]
            },
        }
        :param input_dict: dict like input json object
        :type input_dict: dict
        """
        # Parse datasets info
        datasets_infos = {}
        for dataset in input_dict[DatasetsMapping.DATASETS_INFO_KEY]:
            datasets_infos[dataset] = DatasetInfo.deserialize(
                input_dict=input_dict[DatasetsMapping.DATASETS_INFO_KEY][dataset]
            )

        # Parse namespace datasets mapping
        namespace_datasets_mapping = {}
        input_dict_dataset_mapping = input_dict[DatasetsMapping.NAMESPACE_KEY]
        for namespace in input_dict_dataset_mapping:
            namespace_datasets_mapping[namespace] = []
            for dataset in input_dict_dataset_mapping[namespace]:
                namespace_datasets_mapping[namespace].append(datasets_infos[dataset])
        return DatasetsMapping(
            datasets_infos=datasets_infos,
            namespace_datasets_mapping=namespace_datasets_mapping,
        )

    @staticmethod
    def from_json_file(file_path: str) -> "DatasetsMapping":
        """
        Method to deserialize from a json file
        :param file_path: path of the file to deserialize
        :type file_path: str
        """
        with open(file_path, "rb") as file:
            json_data = json.load(file)
        return DatasetsMapping.deserialize(json_data)
