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
import abc
import logging
from typing import Any, List


class AbstractDatasetsConnector(abc.ABC):
    """
    Abstract class to inherit in order to build specific datasets connector
    """
    __logger = logging.getLogger(__name__)

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

    @abc.abstractmethod
    def get_values_all(self, dataset_identifier: str) -> dict[str:Any]:
        """
        Abstract method to get all values from a dataset for a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        """

    @abc.abstractmethod
    def write_dataset(self, dataset_identifier: str, values_to_write: dict[str:Any], create_if_not_exists:bool=True, override:bool=False) -> None:
        """
        Abstract method to overload in order to write a dataset from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """
    
    def copy_dataset_from(self, connector_from:AbstractDatasetsConnector, dataset_identifier: str, create_if_not_exists:bool=True, override:bool=False):
        """
        Copies a dataset from another AbstractDatasetsConnector
        :param connector_from: Connector to copy dataset from
        :type connector_from: AbstractDatasetsConnector
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """
        self.__logger.debug(f"Copying dataset {dataset_identifier} from {connector_from} to {self}")
        dataset_data = connector_from.get_values_all(dataset_identifier=dataset_identifier)
        self.write_dataset(dataset_identifier=dataset_identifier, values_to_write=dataset_data, create_if_not_exists=create_if_not_exists, override=override)

    def __str__(self) -> str:
        return f"{type(self).__name__}"     


class DatasetGenericException(Exception):
    """
    Generic exception for datasets
    """
    pass

class DatasetNotFoundException(DatasetGenericException):
    """
    Exception when a dataset is not found
    """
    def __init__(self, dataset_name:str):
        self.dataset_name = dataset_name
        super().__init__(f"Dataset '{dataset_name}' not found")

class DatasetUnableToInitializeConnectorException(DatasetGenericException):
    """
    Exception when an error occurs during dataset initialization
    """
    def __init__(self, connector_type:AbstractDatasetsConnector):
        self.connector_type = connector_type
        super().__init__(f"Unable to initialize connector of type {connector_type}")
