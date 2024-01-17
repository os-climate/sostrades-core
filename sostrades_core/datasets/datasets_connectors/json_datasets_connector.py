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
import json
import os
from typing import Any, List

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import AbstractDatasetsConnector


class JSONDatasetsConnector(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in json format
    """

    def __init__(self, file_path: str):
        """
        Constructor for JSON data connector

        :param file_path: file_path for this dataset connector
        :type file_path: str
        """
        super().__init__()
        self.file_path = file_path

    def get_data(self, dataset_identifier: str, data_to_get: List[str]) -> None:
        """
        Method to retrieve data from JSON and fill a data_dict

        :param: dataset_identifier, identifier of the dataset
        :type: string
        """
        # TODO: optimise opening and reading by creating a dedictated abstractDatasetConnector
        json_data = {}
        # Read JSON
        db_path = self.file_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"The connector json file is not found at {db_path}")

        with open(db_path, "r") as file:
            json_data = json.load(file)

        if dataset_identifier not in json_data:
            raise KeyError(f"The dataset {dataset_identifier} is not found in the {self.filename}")

        dataset_data = json_data[dataset_identifier]

        # Filter data
        filtered_data = {key: dataset_data[key] for key in dataset_data if key in data_to_get}
        return filtered_data

    def write_data(self, data_to_write: dict[str:Any]) -> None:
        """
        Method to write data
        """
        raise NotImplementedError()
