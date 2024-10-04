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
from __future__ import annotations

import abc
import logging
from typing import Any

from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo
from sostrades_core.datasets.dataset_info.dataset_info_versions import VERSION_V0


class AbstractDatasetsConnector(abc.ABC):
    """
    Abstract class to inherit in order to build specific datasets connector
    """
    __logger = logging.getLogger(__name__)

    NAME = "name"
    UNIT = "unit"
    DESCRIPTION = "description"
    SOURCE = "source"
    LINK = "link"
    LAST_UPDATE = "last_update_date"
    # generic default metadata dict
    DEFAULT_METADATA_DICT = {
        NAME: "",
        UNIT: "",
        DESCRIPTION: "",
        SOURCE: "",
        LINK: "",
        LAST_UPDATE: ""
    }
    # generic dataset connector value entry (RESERVED)
    VALUE = "value"
    VALUE_KEYS = {VALUE}
    # bigquery specific value entries (RESERVED), need to be defined here to avoid their copy btw connectors as metadata
    PARAMETER_NAME = "parameter_name"
    STRING_VALUE = "parameter_string_value"
    INT_VALUE = "parameter_int_value"
    FLOAT_VALUE = "parameter_float_value"
    BOOL_VALUE = "parameter_bool_value"
    VALUE_KEYS.update({STRING_VALUE, INT_VALUE, FLOAT_VALUE, BOOL_VALUE, PARAMETER_NAME})

    # list of compatible version of dataset info (V0, V1...)
    compatible_dataset_info_version = [VERSION_V0]

    def check_dataset_info_version(self, dataset_identifier: AbstractDatasetInfo):
        """
        Check that the version of the dataset info is compatible with the version of the dataset Connector
        raise an error in case of incompatibility
        :param: dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        """
        if dataset_identifier.version_id not in self.compatible_dataset_info_version:
            raise DatasetDeserializeException(dataset_identifier.dataset_id, f'the version {dataset_identifier.version_id} is not compatible with the dataset connector {dataset_identifier.connector_id}')
        
    
    def get_values(self, dataset_identifier: AbstractDatasetInfo, data_to_get: dict[str:str]) -> dict[str:Any]:
        """
        get a list of data from a specific API
        :param: dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param data_to_get: dict of data name and type of data to get {name: type}
        :type data_to_get: dict[str:str]
        """
        self.check_dataset_info_version(dataset_identifier)

        return self._get_values(dataset_identifier=dataset_identifier,data_to_get=data_to_get)

    @abc.abstractmethod
    def _get_values(self, dataset_identifier: AbstractDatasetInfo, data_to_get: dict[str:str]) -> dict[str:Any]:
        """
        Abstract method to overload in order to get a list of data from a specific API
        :param: dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param data_to_get: dict of data name and type of data to get {name: type}
        :type data_to_get: dict[str:str]
        """

    
    def write_values(self, dataset_identifier: AbstractDatasetInfo, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> dict[str:Any]:
        """
        Abstract method to overload in order to write a data from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._write_values( dataset_identifier, values_to_write, data_types_dict) 
    
    @abc.abstractmethod
    def _write_values(self, dataset_identifier: AbstractDatasetInfo, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> dict[str:Any]:
        """
        Protected Abstract method to overload in order to write a data from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """
        return values_to_write

    
    def get_values_all(self, dataset_identifier: AbstractDatasetInfo, data_types_dict: dict[str:str]) -> dict[str:Any]:
        """
        get all values from a dataset for a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._get_values_all( dataset_identifier, data_types_dict) 

    @abc.abstractmethod
    def _get_values_all(self, dataset_identifier: AbstractDatasetInfo, data_types_dict: dict[str:str]) -> dict[str:Any]:
        """
        Protected Abstract method to get all values from a dataset for a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        """

    @abc.abstractmethod
    def get_datasets_available(self) -> list[AbstractDatasetInfo]:
        """
        Abstract method to get all available datasets for a specific API
        """

    
    def write_dataset(self, dataset_identifier: AbstractDatasetInfo, values_to_write: dict[str:Any], data_types_dict: dict[str:str], create_if_not_exists: bool = True, override: bool = False) -> dict[str:Any]:
        """
        write a dataset from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        :return: values_to_write: dict[str: Any]
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._write_dataset(dataset_identifier, values_to_write, data_types_dict, create_if_not_exists, override) 
    
    @abc.abstractmethod
    def _write_dataset(self, dataset_identifier: AbstractDatasetInfo, values_to_write: dict[str:Any], data_types_dict: dict[str:str], create_if_not_exists: bool = True, override: bool = False) -> dict[str:Any]:
        """
        Protected Abstract method to overload in order to write a dataset from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        :return: values_to_write: dict[str: Any]
        """
        return values_to_write


    def build_path_to_data(self, dataset_identifier:AbstractDatasetInfo, data_name:str, data_type:str)->str:
        """
        build the path to a dataset data for a specific API
        :param dataset_identifier: dataset identifier into connector
        :type dataset_identifier: DatasetInfo
        :param data_name: data in dataset
        :type data_name: str
        :param data_type: type of the data in dataset
        :type data_type: str
        :return: path/url/uri to find the dataset data
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._build_path_to_data(dataset_identifier, data_name, data_type)
    

    def _build_path_to_data(self, dataset_identifier:AbstractDatasetInfo, data_name:str, data_type:str)->str:
        """
        Method that can be overloaded in order to build the path to a dataset data for a specific API
        :param dataset_identifier: dataset identifier into connector
        :type dataset_identifier: DatasetInfo
        :param data_name: data in dataset
        :type data_name: str
        :param data_type: type of the data in dataset
        :type data_type: str
        :return: path/url/uri to find the dataset data
        """
        return ""

    def copy_dataset_from(self, connector_from: AbstractDatasetsConnector, dataset_identifier: AbstractDatasetInfo, data_types_dict: dict[str:str], create_if_not_exists: bool = True, override: bool = False):
        """
        Copies a dataset from another AbstractDatasetsConnector
        :param connector_from: Connector to copy dataset from
        :type connector_from: AbstractDatasetsConnector
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """
        self.check_dataset_info_version(dataset_identifier)
        self.__logger.debug(f"Copying dataset {dataset_identifier.dataset_id} from {connector_from} to {self}")
        dataset_data = connector_from.get_values_all(dataset_identifier=dataset_identifier, data_types_dict=data_types_dict)
        self.write_dataset(dataset_identifier=dataset_identifier, values_to_write=dataset_data, data_types_dict=data_types_dict, create_if_not_exists=create_if_not_exists, override=override)

    def copy_all_datasets_from(self, connector_from: AbstractDatasetsConnector, data_types_dict: dict[str:str], create_if_not_exists: bool = True, override: bool = False):
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

    # Metadata handling
    def _new_datum(self, old_datum):
        if old_datum is None:
            new_datum = self.DEFAULT_METADATA_DICT.copy()
        else:
            new_datum = self._extract_metadata_from_datum(old_datum)
            # new_datum.update({key: value for key, value in self.DEFAULT_METADATA_DICT.items() if key not in new_datum})
        return new_datum

    def _insert_value_into_datum(self,
                                 value: Any,
                                 datum: dict[str:Any],
                                 parameter_name: str,
                                 parameter_type: str) -> dict[str:Any]:
        new_datum = {self.VALUE: value}
        new_datum.update(self._new_datum(datum))
        # TODO: keeping dataset metadata "as is", insert metadata handling here
        return new_datum

    def _update_data_with_values(self, data: dict[str:dict[str:Any]], values: dict[str:Any],
                                 data_types: dict[str:str]) -> None:
        for key, value in values.items():
            new_datum = self._insert_value_into_datum(value=value,
                                                      datum=data.get(key, None),
                                                      parameter_name=key,
                                                      parameter_type=data_types[key])
            if new_datum is not None:
                data[key] = new_datum

    def _extract_value_from_datum(self, datum: dict[str:Any]) -> Any:
        return datum[self.VALUE]

    def _extract_metadata_from_datum(self, datum_to_extract: dict[str:Any]) -> dict[str:str]:
        return {_k: _v for _k, _v in datum_to_extract.items() if _k not in self.VALUE_KEYS}

    def extract_values_from_data(self, data_to_extract: dict[str:dict[str:Any]]) -> dict[str:Any]:
        return {key: self._extract_value_from_datum(_datum) for key, _datum in data_to_extract.items()}

    def extract_metadata_from_data(self, data_to_extract: dict[str:dict[str:Any]]) -> dict[str:dict[str:str]]:
        return {key: self._extract_metadata_from_datum(_datum) for key, _datum in data_to_extract.items()}


class DatasetGenericException(Exception):
    """
    Generic exception for datasets
    """
    pass


class DatasetNotFoundException(DatasetGenericException):
    """
    Exception when a dataset is not found
    """
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        super().__init__(f"Dataset '{dataset_name}' not found")


class DatasetDeserializeException(DatasetGenericException):
    """
    Exception when a dataset deserializing
    """
    def __init__(self, dataset_name: str, error_message: str):
        self.dataset_name = dataset_name
        super().__init__(f"Error reading dataset '{dataset_name}': \n{error_message}")


class DatasetUnableToInitializeConnectorException(DatasetGenericException):
    """
    Exception when an error occurs during dataset initialization
    """
    def __init__(self, connector_type: AbstractDatasetsConnector):
        self.connector_type = connector_type
        super().__init__(f"Unable to initialize connector of type {connector_type}")
