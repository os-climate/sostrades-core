'''
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
'''
from dataclasses import dataclass
from typing import Any

from sostrades_core.datasets.dataset_info import DatasetInfo
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import AbstractDatasetsConnector


@dataclass()
class Dataset:
    """
    Dataset class
    """
    dataset_info: DatasetInfo
    connector: AbstractDatasetsConnector

    def get_values(self, data_dict: dict[str:str]) -> dict[str:Any]:
        """
        Get dataset data and return a data dict with values

        :param data_names: dict of names and types of data to retrieve
        :type data_names: dict[str:str] name: type
        """
        return self.connector.get_values(dataset_identifier=self.dataset_info.dataset_id, data_to_get=data_dict)
