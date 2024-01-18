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
import logging
import os
from typing import Any, List

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import AbstractDatasetsConnector, DatasetGenericException, DatasetNotFoundException


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
        self.__file_path = file_path
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug("Initializing JSON connector")

        # In json, we have to load the full file to retrieve values, so cache it
        self.__json_data = None

    def __load_json_data(self):
        """
        Method to load data from json file
        Populates self.__json_data
        """
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException() from FileNotFoundError(f"The connector json file is not found at {db_path}")

        with open(db_path, "r", encoding="utf-8") as file:
            self.__json_data = json.load(fp=file)

    def __save_json_data(self):
        """
        Method to save data to json file
        """
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException() from FileNotFoundError(f"The connector json file is not found at {db_path}")

        with open(db_path, "w", encoding="utf-8") as file:
            json.dump(obj=self.__json_data, fp=file, indent=4)

    def get_values(self, dataset_identifier: str, data_to_get: List[str]) -> None:
        """
        Method to retrieve data from JSON and fill a data_dict

        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: str

        :param data_to_get: data to retrieve, list of names
        :type data_to_get: List[str]
        """
        self.__logger.debug(f"Getting values {data_to_get} for dataset {dataset_identifier} for connector {self}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__load_json_data()

        if dataset_identifier not in self.__json_data:
            raise DatasetNotFoundException(f"The dataset {dataset_identifier} is not found in the file {self.__file_path}")

        dataset_data = self.__json_data[dataset_identifier]

        # Filter data
        filtered_data = {key: dataset_data[key] for key in dataset_data if key in data_to_get}
        self.__logger.debug(f"Values obtained {list(filtered_data.keys())} for dataset {dataset_identifier} for connector {self}")
        return filtered_data

    def write_values(self, dataset_identifier: str, values_to_write: dict[str:Any]) -> None:
        """
        Method to write data
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: List[str]
        """
        # Read JSON if not read already
        self.__logger.debug(f"Writing values in dataset {dataset_identifier} for connector {self}")
        if self.__json_data is None:
            self.__load_json_data()

        if dataset_identifier not in self.__json_data:
            raise DatasetNotFoundException(f"Dataset {dataset_identifier} not found in json file {self.__file_path}")

        # Write data
        self.__json_data[dataset_identifier].update(values_to_write)

        self.__save_json_data()
    
    def get_values_all(self, dataset_identifier: str) -> dict[str:Any]:
        """
        Abstract method to get all values from a dataset for a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier} for connector {self}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__load_json_data()

        if dataset_identifier not in self.__json_data:
            raise DatasetNotFoundException(f"The dataset {dataset_identifier} is not found in the file {self.__file_path}")

        dataset_data = self.__json_data[dataset_identifier]
        return dataset_data
        

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
        self.__logger.debug(f"Writing dataset {dataset_identifier} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})")
        if dataset_identifier not in self.__json_data:
            # Handle dataset creation
            if create_if_not_exists:
                self.__json_data = {}
            else:
                raise DatasetNotFoundException(dataset_identifier)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier} would be overriden")
        
        self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write)
            