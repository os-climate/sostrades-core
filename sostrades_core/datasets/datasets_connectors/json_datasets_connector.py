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
import json
import logging
import os
from typing import Any

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
    DatasetNotFoundException,
)
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import (
    DatasetSerializerType,
    DatasetsSerializerFactory,
)
from sostrades_core.tools.folder_operations import makedirs_safe


class JSONDatasetsConnector(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in json format
    """

    def __init__(self, file_path: str, create_if_not_exists: bool=False, serializer_type:DatasetSerializerType=DatasetSerializerType.JSON):
        """
        Constructor for JSON data connector


        :param file_path: file_path for this dataset connector
        :type file_path: str
        :param serializer_type: type of serializer to deserialize data from connector
        :type serializer_type: DatasetSerializerType (JSON for jsonDatasetSerializer)
        """
        super().__init__()
        self.__file_path = file_path
        # create file if not exist
        if create_if_not_exists and not os.path.exists(file_path):
            makedirs_safe(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug("Initializing JSON connector")
        self._datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)

        # In json, we have to load the full file to retrieve values, so cache it
        self.__json_data = None

    def __load_json_data(self):
        """
        Method to load data from json file
        Populates self.__json_data
        """
        db_path = self.__file_path
        if not os.path.exists(db_path):
            raise DatasetGenericException(f"The connector json file is not found at {db_path}") from FileNotFoundError()

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

    def get_values(self, dataset_identifier: str, data_group_identifier: str, data_to_get: dict[str:str]) -> dict[str:Any]:
        """
        Method to retrieve data from a single data group within the JSON dataset and fill a data_dict
        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: str
        :param data_group_identifier: identifier of the data group inside the dataset
        :type data_group_identifier: str
        :param data_to_get: data to retrieve, dict of names and types
        :type data_to_get: dict[str:str]
        :return dictionary {parameter_name: value} retreived from the dataset
        """
        self.__logger.debug(f"Getting values {data_to_get.keys()} for data group {data_group_identifier} in dataset "
                            f"{dataset_identifier} with connector {self}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__load_json_data()

        if dataset_identifier not in self.__json_data:
            raise DatasetNotFoundException(dataset_identifier)

        # Filter data
        _group_data = self.__json_data[dataset_identifier][data_group_identifier]
        filtered_values = {key: self._datasets_serializer.convert_from_dataset_data(key,
                                                                                    self._extract_value_from_datum(_group_data[key]),
                                                                                    data_to_get)
                           for key in _group_data if key in data_to_get}
        self.__logger.debug(f"Values obtained {list(filtered_values.keys())} for dataset {dataset_identifier} for connector {self}")
        return filtered_values

    def get_datasets_available(self) -> list[str]:
        """
        Get all available datasets for a specific API
        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__load_json_data()
        return list(self.__json_data.keys())

    def get_data_groups_for_dataset(self, dataset_identifier: str) -> list[str]:
        # FIXME: only implemented here + unused
        """
        Get all available data groups for a specific dataset in a specific API
        """
        self.__logger.debug(f"Getting all data groups for connector {self} for dataset {dataset_identifier}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__load_json_data()
        return list(self.__json_data[dataset_identifier].keys())

    def write_values(self, dataset_identifier: str, data_group_identifier: str, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> dict[str: Any]:
        """
        Method to write data
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str], name, value
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        """
        # FIXME: functionality looks untested !?
        # Read JSON if not read already
        self.__logger.debug(f"Writing values in dataset {dataset_identifier} for connector {self}")
        if self.__json_data is None:
            self.__load_json_data()
        # Impose that dataset exists
        if dataset_identifier not in self.__json_data:
            raise DatasetNotFoundException(dataset_identifier)
        # But allow on-the-go creation of data group
        if data_group_identifier not in self.__json_data[dataset_identifier]:
            self.__json_data[dataset_identifier][data_group_identifier] = {}
        
        # Write data
        dataset_values = {key: self._datasets_serializer.convert_to_dataset_data(key,
                                                                                 value,
                                                                                 data_types_dict)
                          for key, value in values_to_write.items()}
        self._update_data_with_values(self.__json_data[dataset_identifier][data_group_identifier], dataset_values, data_types_dict)
        self.__save_json_data()
        return values_to_write

    def get_values_all(self, dataset_identifier: str, data_types_dict: dict[str:dict[str:str]]) -> dict[str:dict[str:Any]]:
        """
        Abstract method to get all values from a dataset for the JSON connector.
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_types_dict: dict of data type by data group {data_group: {name: type}}
        :type data_types_dict: dict[str:dict[str:str]]
        :return: dataset values by group
        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier} for connector {self}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__load_json_data()

        if dataset_identifier not in self.__json_data:
            raise DatasetNotFoundException(dataset_identifier)

        dataset_values_by_group = dict()
        for _group_id, _group_data in self.__json_data[dataset_identifier].items():
            dataset_values_by_group[_group_id] = {
                key: self._datasets_serializer.convert_from_dataset_data(key,
                                                                         self._extract_value_from_datum(datum),
                                                                         data_types_dict[_group_id])
                for key, datum in _group_data.items()}
        return dataset_values_by_group


    def write_dataset(self, dataset_identifier: str, values_to_write: dict[str:dict[str:Any]], data_types_dict:dict[str:dict[str:str]],
                      create_if_not_exists:bool=True, override:bool=False) -> dict[str: dict[str:Any]]:
        """
        Abstract method to overload in order to write a dataset for the JSON connector.
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {data_group: {parameter_name: value}
        :type values_to_write: dict[str:dict[str:Any]]
        :param data_types_dict: dict of data types {data_group: {parameter_name: type}}
        :type data_types_dict: dict[str:dict[str:str]]
        :param create_if_not_exists: create the dataset if it does not exists (raises otherwise)
        :type create_if_not_exists: bool
        :param override: override dataset if it exists (raises otherwise)
        :type override: bool
        :return: values_to_write
        """
        self.__logger.debug(f"Writing dataset {dataset_identifier} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__load_json_data()

        if dataset_identifier not in self.__json_data:
            # Handle dataset creation
            if create_if_not_exists:
                self.__json_data[dataset_identifier] = {}
            else:
                raise DatasetNotFoundException(dataset_identifier)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier} would be overriden")

        written_values = dict()
        for _group_id, _group_data in values_to_write.items():
            written_values[_group_id] = self.write_values(dataset_identifier=dataset_identifier,
                                                          data_group_identifier=_group_id,
                                                          values_to_write=_group_data,
                                                          data_types_dict=data_types_dict[_group_id])
        return written_values

    def clear_dataset(self, dataset_id: str) -> None:
        """
        Utility method to remove the directory corresponding to a given dataset_id within the JSON database.
        :param dataset_id: identifier of the dataset to be removed
        :type dataset_id: str
        :return: None
        """
        if self.__json_data is None:
            self.__load_json_data()
        if dataset_id in self.__json_data:
            del self.__json_data[dataset_id]
        self.__save_json_data()