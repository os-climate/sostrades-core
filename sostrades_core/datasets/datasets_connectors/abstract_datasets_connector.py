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
from typing import TYPE_CHECKING, Any, Dict, List

from sostrades_core.datasets.dataset_info.dataset_info_versions import VERSION_V0

if TYPE_CHECKING:
    from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo


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

    DATASET_FOLDER = "Dataset folder"
    # list of compatible version of dataset info (V0, V1...)
    COMPATIBLE_DATASET_INFO_VERSION = [VERSION_V0]
    CONNECTOR_ID = "connector_id"
    @property
    def compatible_dataset_info_version(self):
        return set(self.COMPATIBLE_DATASET_INFO_VERSION)


    def check_dataset_info_version(self, dataset_identifier: AbstractDatasetInfo) -> None:
        """
        Check that the version of the dataset info is compatible with the version of the dataset Connector
        raise an error in case of incompatibility

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector

        Raises:
            DatasetDeserializeException: If the version is not compatible
        """
        if dataset_identifier.version_id not in self.compatible_dataset_info_version:
            raise DatasetDeserializeException(dataset_identifier.dataset_id, f'the version {dataset_identifier.version_id} is not compatible with the dataset connector {dataset_identifier.connector_id}')

    def check_connector_compatible_version(self, connector: AbstractDatasetsConnector) -> bool:
        """
        Check that the associated connector is compatible with the current connector version
        raise an error in case of incompatibility

        Args:
            connector (AbstractDatasetsConnector): connector to check

        Returns:
            bool: True if the connectors have at least one version in common
        """
        return bool(set(self.compatible_dataset_info_version) & set(connector.compatible_dataset_info_version))

    def get_values(self, dataset_identifier: AbstractDatasetInfo, data_to_get: Dict[str, str]) -> Dict[str, Any]:
        """
        Get a list of data from a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            data_to_get (Dict[str, str]): dict of data name and type of data to get {name: type}

        Returns:
            Dict[str, Any]: Retrieved data
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._get_values(dataset_identifier=dataset_identifier, data_to_get=data_to_get)

    @abc.abstractmethod
    def _get_values(self, dataset_identifier: AbstractDatasetInfo, data_to_get: Dict[str, str]) -> Dict[str, Any]:
        """
        Abstract method to overload in order to get a list of data from a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            data_to_get (Dict[str, str]): dict of data name and type of data to get {name: type}

        Returns:
            Dict[str, Any]: Retrieved data
        """

    def write_values(self, dataset_identifier: AbstractDatasetInfo, values_to_write: Dict[str, Any], data_types_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Write data to a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            values_to_write (Dict[str, Any]): dict of data to write {name: value}
            data_types_dict (Dict[str, str]): dict of data type {name: type}

        Returns:
            Dict[str, Any]: Written data
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._write_values(dataset_identifier, values_to_write, data_types_dict)

    @abc.abstractmethod
    def _write_values(self, dataset_identifier: AbstractDatasetInfo, values_to_write: Dict[str, Any], data_types_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Protected Abstract method to overload in order to write data to a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            values_to_write (Dict[str, Any]): dict of data to write {name: value}
            data_types_dict (Dict[str, str]): dict of data type {name: type}

        Returns:
            Dict[str, Any]: Written data
        """
        return values_to_write

    def get_values_all(self, dataset_identifier: AbstractDatasetInfo, data_types_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Get all values from a dataset for a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            data_types_dict (Dict[str, str]): dict of data types {name: type}

        Returns:
            Dict[str, Any]: All values from the dataset
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._get_values_all(dataset_identifier, data_types_dict)

    @abc.abstractmethod
    def _get_values_all(self, dataset_identifier: AbstractDatasetInfo, data_types_dict: Dict[str, str]) -> Dict[str, Any]:
        """
        Protected Abstract method to get all values from a dataset for a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            data_types_dict (Dict[str, str]): dict of data types {name: type}

        Returns:
            Dict[str, Any]: All values from the dataset
        """

    @abc.abstractmethod
    def get_datasets_available(self) -> List[AbstractDatasetInfo]:
        """
        Abstract method to get all available datasets for a specific API

        Returns:
            List[AbstractDatasetInfo]: List of available datasets
        """

    def write_dataset(self, dataset_identifier: AbstractDatasetInfo, values_to_write: Dict[str, Any], data_types_dict: Dict[str, str], create_if_not_exists: bool = True, override: bool = False) -> Dict[str, Any]:
        """
        Write a dataset to a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            values_to_write (Dict[str, Any]): dict of data to write {name: value}
            data_types_dict (Dict[str, str]): dict of data types {name: type}
            create_if_not_exists (bool, optional): Create the dataset if it does not exist. Defaults to True.
            override (bool, optional): Override dataset if it exists. Defaults to False.

        Returns:
            Dict[str, Any]: Written data
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._write_dataset(dataset_identifier, values_to_write, data_types_dict, create_if_not_exists, override)

    @abc.abstractmethod
    def _write_dataset(self, dataset_identifier: AbstractDatasetInfo, values_to_write: Dict[str, Any], data_types_dict: Dict[str, str], create_if_not_exists: bool = True, override: bool = False) -> Dict[str, Any]:
        """
        Protected Abstract method to overload in order to write a dataset to a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            values_to_write (Dict[str, Any]): dict of data to write {name: value}
            data_types_dict (Dict[str, str]): dict of data types {name: type}
            create_if_not_exists (bool, optional): Create the dataset if it does not exist. Defaults to True.
            override (bool, optional): Override dataset if it exists. Defaults to False.

        Returns:
            Dict[str, Any]: Written data
        """
        return values_to_write

    def build_path_to_data(self, dataset_identifier: AbstractDatasetInfo, data_name: str, data_type: str) -> str:
        """
        Build the path to a dataset data for a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier into connector
            data_name (str): data in dataset
            data_type (str): type of the data in dataset

        Returns:
            str: Path/URL/URI to find the dataset data
        """
        self.check_dataset_info_version(dataset_identifier)
        return self._build_path_to_data(dataset_identifier, data_name, data_type)

    def _build_path_to_data(self, dataset_identifier: AbstractDatasetInfo, data_name: str, data_type: str) -> str:
        """
        Method that can be overloaded in order to build the path to a dataset data for a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): dataset identifier into connector
            data_name (str): data in dataset
            data_type (str): type of the data in dataset

        Returns:
            str: Path/URL/URI to find the dataset data
        """
        return ""

    def copy_dataset_from(self, connector_from: AbstractDatasetsConnector, dataset_identifier: AbstractDatasetInfo, data_types_dict: Dict[str, str], create_if_not_exists: bool = True, override: bool = False) -> None:
        """
        Copies a dataset from another AbstractDatasetsConnector

        Args:
            connector_from (AbstractDatasetsConnector): Connector to copy dataset from
            dataset_identifier (AbstractDatasetInfo): dataset identifier for connector
            data_types_dict (Dict[str, str]): dict of data types {name: type}
            create_if_not_exists (bool, optional): Create the dataset if it does not exist. Defaults to True.
            override (bool, optional): Override dataset if it exists. Defaults to False.

        Raises:
            DatasetGenericException: If connectors have incompatible versions
        """
        self.check_dataset_info_version(dataset_identifier)
        if not self.check_connector_compatible_version(connector_from):
            raise DatasetGenericException("Connectors have incompatible versions, for now it is not possible to copy data V0 <-> V1")

        self.__logger.debug(f"Copying dataset {dataset_identifier.dataset_id} from {connector_from} to {self}")
        dataset_data = connector_from.get_values_all(dataset_identifier=dataset_identifier, data_types_dict=data_types_dict)
        self.write_dataset(dataset_identifier=dataset_identifier, values_to_write=dataset_data, data_types_dict=data_types_dict, create_if_not_exists=create_if_not_exists, override=override)

    def copy_all_datasets_from(self, connector_from: AbstractDatasetsConnector, data_types_dict: Dict[str, str], create_if_not_exists: bool = True, override: bool = False) -> None:
        """
        Copies all datasets from another AbstractDatasetsConnector

        Args:
            connector_from (AbstractDatasetsConnector): Connector to copy dataset from
            data_types_dict (Dict[str, str]): dict of data types {name: type}
            create_if_not_exists (bool, optional): Create the dataset if it does not exist. Defaults to True.
            override (bool, optional): Override dataset if it exists. Defaults to False.

        Raises:
            DatasetGenericException: If connectors have incompatible versions
        """
        if not self.check_connector_compatible_version(connector_from):
            raise DatasetGenericException("Connectors have incompatible versions, for now it is not possible to copy data V0 <-> V1")

        self.__logger.debug(f"Copying all datasets from {connector_from} to {self}")
        datasets = connector_from.get_datasets_available()
        for dataset_identifier in datasets:
            dataset_data = connector_from.get_values_all(dataset_identifier=dataset_identifier, data_types_dict=data_types_dict)
            self.write_dataset(dataset_identifier=dataset_identifier, values_to_write=dataset_data, data_types_dict=data_types_dict, create_if_not_exists=create_if_not_exists, override=override)

    def __str__(self) -> str:
        return f"{type(self).__name__}"

    # CLEARING
    def clear_dataset(self, dataset_id: str) -> None:
        """
        Optional utility method to remove a given dataset_id within a certain connector.

        Args:
            dataset_id (str): Identifier of the dataset to be removed
        """
        raise NotImplementedError

    def clear_all_datasets(self):
        """
        Optional utility method to remove all datasets in a connector.
        """
        map(lambda _d: self.clear_dataset(_d.dataset_id), self.get_datasets_available())

    def clear_connector(self):
        """
        Optional utility method to completely clear a connector further than clearing all datasets, if it applies, e.g.
        by deleting the root directory of a local connector, or by deleting the database file of a json connector. It
        defaults to clear_all_datasets unless overloaded.
        """
        self.clear_all_datasets()

    # METADATA HANDLING
    def _new_datum(self, old_datum: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new datum with default metadata

        Args:
            old_datum (Dict[str, Any]): Old datum

        Returns:
            Dict[str, Any]: New datum with default metadata
        """
        if old_datum is None:
            new_datum = self.DEFAULT_METADATA_DICT.copy()
        else:
            new_datum = self._extract_metadata_from_datum(old_datum)
        return new_datum

    def _insert_value_into_datum(self, value: Any, datum: Dict[str, Any], parameter_name: str, parameter_type: str) -> Dict[str, Any]:
        """
        Insert value into datum

        Args:
            value (Any): Value to insert
            datum (Dict[str, Any]): Datum to insert into
            parameter_name (str): Parameter name
            parameter_type (str): Parameter type

        Returns:
            Dict[str, Any]: Updated datum
        """
        new_datum = {self.VALUE: value}
        new_datum.update(self._new_datum(datum))
        return new_datum

    def _update_data_with_values(self, data: Dict[str, Dict[str, Any]], values: Dict[str, Any], data_types: Dict[str, str]) -> None:
        """
        Update data with values

        Args:
            data (Dict[str, Dict[str, Any]]): Data to update
            values (Dict[str, Any]): Values to update with
            data_types (Dict[str, str]): Data types
        """
        for key, value in values.items():
            new_datum = self._insert_value_into_datum(value=value, datum=data.get(key, None), parameter_name=key, parameter_type=data_types[key])
            if new_datum is not None:
                data[key] = new_datum

    def _extract_value_from_datum(self, datum: Dict[str, Any]) -> Any:
        """
        Extract value from datum

        Args:
            datum (Dict[str, Any]): Datum to extract from

        Returns:
            Any: Extracted value
        """
        return datum[self.VALUE]

    def _extract_metadata_from_datum(self, datum_to_extract: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract metadata from datum

        Args:
            datum_to_extract (Dict[str, Any]): Datum to extract from

        Returns:
            Dict[str, str]: Extracted metadata
        """
        return {_k: _v for _k, _v in datum_to_extract.items() if _k not in self.VALUE_KEYS}

    def extract_values_from_data(self, data_to_extract: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract values from data

        Args:
            data_to_extract (Dict[str, Dict[str, Any]]): Data to extract from

        Returns:
            Dict[str, Any]: Extracted values
        """
        return {key: self._extract_value_from_datum(_datum) for key, _datum in data_to_extract.items()}

    def extract_metadata_from_data(self, data_to_extract: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """
        Extract metadata from data

        Args:
            data_to_extract (Dict[str, Dict[str, Any]]): Data to extract from

        Returns:
            Dict[str, Dict[str, str]]: Extracted metadata
        """
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
