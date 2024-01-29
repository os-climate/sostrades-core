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
import numpy as np

import pandas as pd


class AbstractDatasetsConnector(abc.ABC):
    """
    Abstract class to inherit in order to build specific datasets connector
    """
    __logger = logging.getLogger(__name__)

    @abc.abstractmethod
    def get_values(self, dataset_identifier: str, data_to_get: dict[str:str]) -> dict[str:Any]:
        """
        Abstract method to overload in order to get a list of data from a specific API
        :param: dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_to_get: dict of data name and type of data to get {name: type}
        :type data_to_get: dict[str:str]
        """

    @abc.abstractmethod
    def write_values(self, dataset_identifier: str, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> None:
        """
        Abstract method to overload in order to write a data from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """

    @abc.abstractmethod
    def get_values_all(self, dataset_identifier: str, data_types_dict:dict[str:str]) -> dict[str:Any]:
        """
        Abstract method to get all values from a dataset for a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        """

    @abc.abstractmethod
    def get_datasets_available(self) -> list[str]:
        """
        Abstract method to get all available datasets for a specific API
        """

    @abc.abstractmethod
    def write_dataset(self, dataset_identifier: str, values_to_write: dict[str:Any], data_types_dict:dict[str:str], create_if_not_exists:bool=True, override:bool=False) -> None:
        """
        Abstract method to overload in order to write a dataset from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """

    def convert_from_connector_data(self, data_name:str, data_value:Any, data_types_dict:dict[str:str])-> Any:
        '''
        Convert data_value into data_type from the connector
        To be override for specific conversion.
        This function convert dataframe into dict and arrays into list, other types doesn't move.
        Can be used for json mapping for example.
        :param data_name: name of the data that is converted
        :type data_name: str
        :param data_value: value of the data that is converted
        :type data_value: Any
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        '''
        # retreive the type of the data into the data_type_dict. 
        # If the data type os not found, the data value is not converted
        data_type = None
        if data_name in data_types_dict.keys():
            data_type = data_types_dict[data_name]

        converted_data = ""
        try:
            if data_type in ['string', 'int', 'float', 'bool', 'list', 'dict']:
                converted_data = data_value
            elif data_type == 'dataframe':
                converted_data = pd.DataFrame.from_dict(data_value)
            elif data_type == 'array':
                converted_data = np.array(data_value)
            else:
                converted_data = data_value
                self.__logger.warning(f"Data type {data_type} for data {data_name} not found in default type list 'string', 'int', 'float', 'bool', 'list', 'dict', 'dataframe, 'array'.")
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")

        return converted_data
    
    def convert_to_connector_data(self, data_name:str, data_value:Any, data_types_dict:dict[str:str])-> Any:
        '''
        Convert data_value into connector format
        :param data_name: name of the data that is converted
        :type data_name: str
        :param data_value: value of the data that is converted
        :type data_value: Any
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        '''
        # retreive the type of the data into the data_type_dict. 
        # If the data type os not found, the data value is not converted
        data_type = None
        if data_name in data_types_dict.keys():
            data_type = data_types_dict[data_name]

        converted_data = ""
        try:
            if data_type in ['string', 'int', 'float', 'bool', 'list', 'dict']:
                converted_data = data_value
            elif data_type == 'dataframe':
                # convert dataframe into dict with orient='list' to have {column:values}
                converted_data = pd.DataFrame.to_dict(data_value,'list')
            elif data_type == 'array':
                converted_data = list(data_value)
            else:
                converted_data = data_value
                self.__logger.warning(f"Data type {data_type} for data {data_name} not found in default type list 'string', 'int', 'float', 'bool', 'list', 'dict', 'dataframe, 'array'.")
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")

        return converted_data
    
    
    def copy_dataset_from(self, connector_from:AbstractDatasetsConnector, dataset_identifier: str, data_types_dict:dict[str:str], create_if_not_exists:bool=True, override:bool=False):
        """
        Copies a dataset from another AbstractDatasetsConnector
        :param connector_from: Connector to copy dataset from
        :type connector_from: AbstractDatasetsConnector
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """
        self.__logger.debug(f"Copying dataset {dataset_identifier} from {connector_from} to {self}")
        dataset_data = connector_from.get_values_all(dataset_identifier=dataset_identifier, data_types_dict=data_types_dict)
        self.write_dataset(dataset_identifier=dataset_identifier, values_to_write=dataset_data, data_types_dict=data_types_dict, create_if_not_exists=create_if_not_exists, override=override)

    
    def copy_all_datasets_from(self, connector_from:AbstractDatasetsConnector, data_types_dict:dict[str:str], create_if_not_exists:bool=True, override:bool=False):
        """
        Copies all datasets from another AbstractDatasetsConnector
        :param connector_from: Connector to copy dataset from
        :type connector_from: AbstractDatasetsConnector
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """
        self.__logger.debug(f"Copying all datasets from {connector_from} to {self}")
        datasets = connector_from.get_datasets_available()
        for dataset_identifier in datasets:
            dataset_data = connector_from.get_values_all(dataset_identifier=dataset_identifier)
            self.write_dataset(dataset_identifier=dataset_identifier, values_to_write=dataset_data, data_types_dict=data_types_dict, create_if_not_exists=create_if_not_exists, override=override)

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
