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

import abc
from typing import Any, List


class AbstractDatasetsConnector(abc.ABC):
    """
    Abstract class to inherit in order to build specific datasets connector
    """

    @abc.abstractmethod
    def get_values(self, dataset_identifier: str, data_to_get: List[str]) -> dict[str:Any]:
        """
        Abstract method to overload in order to get a list of data from a specific API
        :param: dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_to_get: list of data name to get
        :type data_to_get: List[str]
        """

    @abc.abstractmethod
    def write_values(self, dataset_identifier: str, values_to_write: dict[str:Any]) -> None:
        """
        Abstract method to overload in order to write a data from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        """
