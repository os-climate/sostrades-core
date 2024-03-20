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
import logging
import os
from typing import Any, Tuple

import pickle

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
    DatasetNotFoundException,
)


class SoSPickleDatasetsConnector(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in pickle format
    """
    VALUE_STR = "value"
    SOS_NS_SEPARATOR = "."

    def __init__(self, file_path: str):
        """
        Constructor for pickle data connector

        :param file_path: file_path for this dataset connector
        :type file_path: str
        """
        super().__init__()
        self.__file_path = file_path
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug("Initializing Pickle connector")

        # In pickle, we have to load the full file to retrieve values, so cache it
        self.__pickle_data = None

    @classmethod
    def __get_pickle_key(cls, data_tag:str, dataset_id:str) -> str:
        """
        Gets the key in pickle from a data name and a dataset id

        :param data_tag: identifier of the dataset
        :type data_tag: str

        :param dataset_id: name of dataset
        :type dataset_id: str
        """
        return dataset_id + SoSPickleDatasetsConnector.SOS_NS_SEPARATOR + data_tag

    @classmethod
    def __get_dataset_id_and_data_name(cls, name_in_pickle:str) -> Tuple[str, str]:
        """
        Gets the dataset id and data name from the name of a variable in pickle

        :param name_in_pickle: name of the parameter in pickle
        :type name_in_pickle: str
        """
        spl = name_in_pickle.split(SoSPickleDatasetsConnector.SOS_NS_SEPARATOR)
        data_name = spl[-1]
        dataset_id = SoSPickleDatasetsConnector.SOS_NS_SEPARATOR.join(spl[:-1])
        return dataset_id, data_name

    def __load_pickle_data(self):
        """
        Method to load data from pickle file
        Populates self.__pickle_data
        """
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException(f"The connector pickle file is not found at {db_path}") from FileNotFoundError()

        with open(db_path, "rb") as file:
            self.__pickle_data = pickle.load(file=file)

    def __save_pickle_data(self):
        """
        Method to save data to pickle file
        """
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException() from FileNotFoundError(f"The connector pickle file is not found at {db_path}")

        with open(db_path, "wb") as file:
            pickle.dump(obj=self.__pickle_data, file=file, indent=4)

    def __has_dataset(self, dataset_id:str):
        """
        Method to check if the pickle file contains dataset
        
        :param dataset_id: data to retrieve, list of names
        :type dataset_id: List[str]
        """
        
        return dataset_id in self.get_datasets_available()

    def get_values(self, dataset_identifier: str, data_to_get: dict[str:str]) -> None:
        """
        Method to retrieve data from pickle and fill a data_dict

        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: str

        :param data_to_get: data to retrieve, dict of names and types
        :type data_to_get: dict[str:str]
        """
        self.__logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier} for connector {self}")
        # Read pickle if not read already
        if self.__pickle_data is None:
            self.__load_pickle_data()

        if not self.__has_dataset(dataset_identifier):
            raise DatasetNotFoundException(dataset_identifier)

        datasets_data = self.__pickle_data

        # Filter data
        filtered_data = {key: datasets_data[self.__get_pickle_key(key, dataset_identifier)][SoSPickleDatasetsConnector.VALUE_STR] for key in datasets_data if key in data_to_get.keys()}
        self.__logger.debug(f"Values obtained {list(filtered_data.keys())} for dataset {dataset_identifier} for connector {self}")
        return filtered_data

    def write_values(self, dataset_identifier: str, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> None:
        """
        Method to write data
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: Dict[str:Any]
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """
        # Read pickle if not read already
        self.__logger.debug(f"Writing values in dataset {dataset_identifier} for connector {self}")
        if self.__pickle_data is None:
            self.__load_pickle_data()

        if not self.__has_dataset(dataset_identifier):
            raise DatasetNotFoundException(dataset_identifier)

        # Perform key mapping
        data_to_update_dict = {self.__get_pickle_key(key, dataset_identifier): value for key, value in values_to_write.items()}

        # Write data
        self.__pickle_data.update(data_to_update_dict)

        self.__save_pickle_data()
    
    def get_values_all(self, dataset_identifier: str) -> dict[str:Any]:
        """
        Abstract method to get all values from a dataset for a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier} for connector {self}")
        # Read pickle if not read already
        if self.__pickle_data is None:
            self.__load_pickle_data()

        if not self.__has_dataset(dataset_identifier):
            raise DatasetNotFoundException(dataset_identifier)
        
        dataset_keys = []
        for key in self.__pickle_data:
            dataset_id, data_name = self.__get_dataset_id_and_data_name(key)
            if dataset_id == dataset_identifier:
                dataset_keys.append(key)

        dataset_data = {key.split(SoSPickleDatasetsConnector.SOS_NS_SEPARATOR)[-1]:self.__pickle_data[key][SoSPickleDatasetsConnector.VALUE_STR] for key in self.__pickle_data if key in dataset_keys}
        return dataset_data
    
    def get_datasets_available(self) -> list[str]:
        """
        Get all available datasets for a specific API
        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        # Read pickle if not read already
        if self.__pickle_data is None:
            self.__load_pickle_data()
        return list(self.__get_dataset_id_and_data_name(key)[0] for key in self.__pickle_data)
        
    def write_dataset(self, dataset_identifier: str, values_to_write: dict[str:Any], data_types_dict:dict[str:str], create_if_not_exists:bool=True, override:bool=False) -> None:
        """
        Abstract method to overload in order to write a dataset from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str:Any]
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:Any]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        """
        self.__logger.debug(f"Writing dataset {dataset_identifier} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})")
        if not self.__has_dataset(dataset_identifier):
            # Handle dataset creation
            if create_if_not_exists:
                # Nothing to do here
                pass
            else:
                raise DatasetNotFoundException(dataset_identifier)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier} would be overriden")
        
        self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write, data_types_dict=data_types_dict)

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "uc1_test_damage_ggo.pickle")
    connector = SoSPickleDatasetsConnector(file_path=file_path)
    print(connector.get_values_all("<study_ph>2.Macroeconomics").keys())
