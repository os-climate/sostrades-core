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

import logging
import os
import pickle
from typing import TYPE_CHECKING, Any, Tuple

from sostrades_core.datasets.dataset_info.dataset_info_v0 import DatasetInfoV0
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
    DatasetNotFoundException,
)

if TYPE_CHECKING:
    from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo


class SoSPickleDatasetsConnector(AbstractDatasetsConnector):
    """Specific dataset connector for dataset in pickle format"""

    VALUE_STR = "value"
    SOS_NS_SEPARATOR = "."

    def __init__(self, connector_id: str, file_path: str) -> None:
        """
        Constructor for pickle data connector

        Args:
            connector_id (str): Identifier for the connector
            file_path (str): File path for this dataset connector

        """
        super().__init__()
        self.__file_path = file_path
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug("Initializing Pickle connector")

        # In pickle, we have to load the full file to retrieve values, so cache it
        self.__pickle_data = None
        self.connector_id = connector_id

    @classmethod
    def __get_pickle_key(cls, data_tag: str, dataset_id: str) -> str:
        """
        Gets the key in pickle from a data name and a dataset id

        Args:
            data_tag (str): Identifier of the dataset
            dataset_id (str): Name of dataset

        Returns:
            str: Key in pickle

        """
        return dataset_id + SoSPickleDatasetsConnector.SOS_NS_SEPARATOR + data_tag

    @classmethod
    def __get_dataset_id_and_data_name(cls, name_in_pickle: str) -> Tuple[str, str]:
        """
        Gets the dataset id and data name from the name of a variable in pickle

        Args:
            name_in_pickle (str): Name of the parameter in pickle

        Returns:
            Tuple[str, str]: Dataset id and data name

        """
        spl = name_in_pickle.split(SoSPickleDatasetsConnector.SOS_NS_SEPARATOR)
        data_name = spl[-1]
        dataset_id = SoSPickleDatasetsConnector.SOS_NS_SEPARATOR.join(spl[:-1])
        return dataset_id, data_name

    def __load_pickle_data(self) -> None:
        """
        Method to load data from pickle file
        Populates self.__pickle_data
        """
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException(f"The connector pickle file is not found at {db_path}") from FileNotFoundError()

        with open(db_path, "rb") as file:
            self.__pickle_data = pickle.load(file=file)

    def __save_pickle_data(self) -> None:
        """Method to save data to pickle file"""
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException() from FileNotFoundError(f"The connector pickle file is not found at {db_path}")

        with open(db_path, "wb") as file:
            pickle.dump(obj=self.__pickle_data, file=file)

    def __has_dataset(self, dataset_id: str) -> bool:
        """
        Method to check if the pickle file contains dataset

        Args:
            dataset_id (str): Dataset identifier

        Returns:
            bool: True if dataset exists, False otherwise

        """
        return dataset_id in self.get_datasets_available()

    def _get_values(self, dataset_identifier: AbstractDatasetInfo, data_to_get: dict[str, str]) -> dict[str, Any]:
        """
        Method to retrieve data from pickle and fill a data_dict

        Args:
            dataset_identifier (AbstractDatasetInfo): Identifier of the dataset
            data_to_get (dict[str, str]): Data to retrieve, dict of names and types

        Returns:
            dict[str, Any]: Retrieved data

        """
        self.__logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier.dataset_id} for connector {self}")
        # Read pickle if not read already
        if self.__pickle_data is None:
            self.__load_pickle_data()

        if not self.__has_dataset(dataset_identifier.dataset_id):
            raise DatasetNotFoundException(dataset_identifier.dataset_id)

        datasets_data = self.__pickle_data

        # Filter data
        filtered_data = {key: datasets_data[self.__get_pickle_key(key, dataset_identifier.dataset_id)][SoSPickleDatasetsConnector.VALUE_STR] for key in datasets_data if key in data_to_get.keys()}
        self.__logger.debug(f"Values obtained {list(filtered_data.keys())} for dataset {dataset_identifier.dataset_id} for connector {self}")
        return filtered_data

    def _write_values(self, dataset_identifier: AbstractDatasetInfo, values_to_write: dict[str, Any], data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Method to write data

        Args:
            dataset_identifier (AbstractDatasetInfo): Dataset identifier for connector
            values_to_write (dict[str, Any]): Dict of data to write {name: value}
            data_types_dict (dict[str, str]): Dict of data type {name: type}

        Returns:
            dict[str, Any]: Written data

        """
        # Read pickle if not read already
        self.__logger.debug(f"Writing values in dataset {dataset_identifier.dataset_id} for connector {self}")
        if self.__pickle_data is None:
            self.__load_pickle_data()

        if not self.__has_dataset(dataset_identifier.dataset_id):
            raise DatasetNotFoundException(dataset_identifier.dataset_id)

        # Perform key mapping
        data_to_update_dict = {self.__get_pickle_key(key, dataset_identifier.dataset_id): value for key, value in values_to_write.items()}

        # Write data
        self.__pickle_data.update(data_to_update_dict)
        self.__save_pickle_data()
        return values_to_write

    def _get_values_all(self, dataset_identifier: AbstractDatasetInfo) -> dict[str, Any]:
        """
        Abstract method to get all values from a dataset for a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): Dataset identifier for connector

        Returns:
            dict[str, Any]: All values from the dataset

        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier.dataset_id} for connector {self}")
        # Read pickle if not read already
        if self.__pickle_data is None:
            self.__load_pickle_data()

        if not self.__has_dataset(dataset_identifier.dataset_id):
            raise DatasetNotFoundException(dataset_identifier.dataset_id)

        dataset_keys = []
        for key in self.__pickle_data:
            dataset_id, _ = self.__get_dataset_id_and_data_name(key)
            if dataset_id == dataset_identifier.dataset_id:
                dataset_keys.append(key)

        dataset_data = {key.split(SoSPickleDatasetsConnector.SOS_NS_SEPARATOR)[-1]: self.__pickle_data[key][SoSPickleDatasetsConnector.VALUE_STR] for key in self.__pickle_data if key in dataset_keys}
        return dataset_data

    def get_datasets_available(self) -> list[AbstractDatasetInfo]:
        """
        Get all available datasets for a specific API

        Returns:
            list[AbstractDatasetInfo]: List of available datasets

        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        # Read pickle if not read already
        if self.__pickle_data is None:
            self.__load_pickle_data()
        return [DatasetInfoV0(self.connector_id, dataset_id) for dataset_id in list(self.__get_dataset_id_and_data_name(key)[0] for key in self.__pickle_data)]

    def _write_dataset(self, dataset_identifier: AbstractDatasetInfo, values_to_write: dict[str, Any], data_types_dict: dict[str, str], create_if_not_exists: bool = True, override: bool = False) -> dict[str, Any]:
        """
        Abstract method to overload in order to write a dataset from a specific API

        Args:
            dataset_identifier (AbstractDatasetInfo): Dataset identifier for connector
            values_to_write (dict[str, Any]): Dict of data to write {name: value}
            data_types_dict (dict[str, str]): Dict of data types {name: type}
            create_if_not_exists (bool, optional): Create the dataset if it does not exists (raises otherwise). Defaults to True.
            override (bool, optional): Override dataset if it exists (raises otherwise). Defaults to False.

        Returns:
            dict[str, Any]: Written data

        """
        self.__logger.debug(f"Writing dataset {dataset_identifier.dataset_id} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})")
        if not self.__has_dataset(dataset_identifier.dataset_id):
            # Handle dataset creation
            if create_if_not_exists:
                # Nothing to do here
                pass
            else:
                raise DatasetNotFoundException(dataset_identifier.dataset_id)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier.dataset_id} would be overriden")

        return self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write, data_types_dict=data_types_dict)
