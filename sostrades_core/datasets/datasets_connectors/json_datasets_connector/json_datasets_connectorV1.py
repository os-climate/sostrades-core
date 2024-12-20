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
from typing import Any

from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo
from sostrades_core.datasets.dataset_info.dataset_info_v1 import DatasetInfoV1
from sostrades_core.datasets.dataset_info.dataset_info_versions import VERSION_V1
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
    DatasetNotFoundException,
)
from sostrades_core.datasets.datasets_connectors.json_datasets_connector.json_dataset_connector_tools import (
    JSONDatasetsConnectorTools,
)
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import (
    DatasetSerializerType,
    DatasetsSerializerFactory,
)


class JSONDatasetsConnectorV1(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in json format
    """

    def __init__(self, connector_id: str, file_path: str, create_if_not_exists: bool = False, serializer_type: DatasetSerializerType = DatasetSerializerType.JSON):
        """
        Constructor for JSON data connector

        Args:
            connector_id (str): Connector identifier
            file_path (str): File path for this dataset connector
            create_if_not_exists (bool, optional): Create file if not exists. Defaults to False.
            serializer_type (DatasetSerializerType, optional): Type of serializer to deserialize data from connector. Defaults to DatasetSerializerType.JSON.
        """
        super().__init__()
        self.__json_tooling = JSONDatasetsConnectorTools(file_path=file_path, create_if_not_exists=create_if_not_exists)
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug("Initializing JSON connector")
        self._datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)
        self.connector_id = connector_id
        self.compatible_dataset_info_version = [VERSION_V1]

        # In json, we have to load the full file to retrieve values, so cache it
        self.__json_data = None

    def get_datasets_available(self) -> list[AbstractDatasetInfo]:
        """
        Get all available datasets for a specific API

        Returns:
            list[AbstractDatasetInfo]: List of available datasets
        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__json_data = self.__json_tooling.load_json_data()

        return [DatasetInfoV1(self.connector_id, dataset_id, group_id) for dataset_id, groups in list(self.__json_data.items()) for group_id in list(groups.keys())]

    def _get_values(self, dataset_identifier: DatasetInfoV1, data_to_get: dict[str, str]) -> dict[str, Any]:
        """
        Method to retrieve data from JSON and fill a data_dict

        Args:
            dataset_identifier (DatasetInfoV1): Identifier of the dataset
            data_to_get (dict[str, str]): Data to retrieve, dict of names and types

        Returns:
            dict[str, Any]: Retrieved data

        Raises:
            DatasetNotFoundException: If dataset or group is not found
        """
        self.__logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier} for connector {self}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__json_data = self.__json_tooling.load_json_data()

        if dataset_identifier.dataset_id not in self.__json_data:
            raise DatasetNotFoundException(dataset_identifier.dataset_id)

        # Filter dataset
        dataset_data = self.__json_data[dataset_identifier.dataset_id]

        # Check group exists
        if dataset_identifier.group_id not in dataset_data:
            raise DatasetNotFoundException(f'group {dataset_identifier.group_id} in dataset {dataset_identifier.dataset_id}')

        dataset_group = dataset_data[dataset_identifier.group_id]
        filtered_values = {key: self._datasets_serializer.convert_from_dataset_data(key,
                                                                                    self._extract_value_from_datum(dataset_group[key]),
                                                                                    data_to_get)
                           for key in dataset_group if key in data_to_get}
        self.__logger.debug(f"Values obtained {list(filtered_values.keys())} for dataset {dataset_identifier.dataset_id}, group {dataset_identifier.group_id} for connector {self}")
        return filtered_values

    def _write_values(self, dataset_identifier: DatasetInfoV1, values_to_write: dict[str, Any], data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Method to write data

        Args:
            dataset_identifier (DatasetInfoV1): Dataset identifier for connector
            values_to_write (dict[str, Any]): Dict of data to write {name: value}
            data_types_dict (dict[str, str]): Dict of data type {name: type}

        Returns:
            dict[str, Any]: Written values

        Raises:
            DatasetNotFoundException: If dataset is not found
        """
        # Read JSON if not read already
        self.__logger.debug(f"Writing values in dataset {dataset_identifier.dataset_id} for connector {self}")
        if self.__json_data is None:
            self.__json_data = self.__json_tooling.load_json_data()

        if dataset_identifier.dataset_id not in self.__json_data:
            raise DatasetNotFoundException(dataset_identifier.dataset_id)

        # Write data
        dataset_values = {key: self._datasets_serializer.convert_to_dataset_data(key,
                                                                                 value,
                                                                                 data_types_dict)
                          for key, value in values_to_write.items()}

        self._update_data_with_values(self.__json_data[dataset_identifier.dataset_id][dataset_identifier.group_id], dataset_values, data_types_dict)
        self.__json_tooling.save_json_data(self.__json_data)
        return values_to_write

    def _get_values_all(self, dataset_identifier: DatasetInfoV1, data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Abstract method to get all values from a dataset for a specific API

        Args:
            dataset_identifier (DatasetInfoV1): Dataset identifier for connector
            data_types_dict (dict[str, str]): Dict of data type {name: type}

        Returns:
            dict[str, Any]: All values from the dataset

        Raises:
            DatasetNotFoundException: If dataset or group is not found
        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier.dataset_id} for connector {self}")
        # Read JSON if not read already
        if self.__json_data is None:
            self.__json_data = self.__json_tooling.load_json_data()

        if dataset_identifier.dataset_id not in self.__json_data:
            raise DatasetNotFoundException(dataset_identifier.dataset_id)

        # Check group exists
        if dataset_identifier.group_id not in self.__json_data[dataset_identifier.dataset_id]:
            raise DatasetNotFoundException(f'group {dataset_identifier.group_id} in dataset {dataset_identifier.dataset_id}')

        dataset_group = self.__json_data[dataset_identifier.dataset_id][dataset_identifier.group_id]

        dataset_values = {key: self._datasets_serializer.convert_from_dataset_data(key,
                                                                                   self._extract_value_from_datum(datum),
                                                                                   data_types_dict)
                          for key, datum in dataset_group.items()}
        return dataset_values

    def _write_dataset(self, dataset_identifier: DatasetInfoV1, values_to_write: dict[str, Any], data_types_dict: dict[str, str], create_if_not_exists: bool = True, override: bool = False) -> dict[str, Any]:
        """
        Abstract method to overload in order to write a dataset from a specific API

        Args:
            dataset_identifier (DatasetInfoV1): Dataset identifier for connector
            values_to_write (dict[str, Any]): Dict of data to write {name: value}
            data_types_dict (dict[str, str]): Dict of data types {name: type}
            create_if_not_exists (bool, optional): Create the dataset if it does not exists (raises otherwise). Defaults to True.
            override (bool, optional): Override dataset if it exists (raises otherwise). Defaults to False.

        Returns:
            dict[str, Any]: Written values

        Raises:
            DatasetNotFoundException: If dataset or group is not found
            DatasetGenericException: If dataset would be overridden
        """
        self.__logger.debug(f"Writing dataset {dataset_identifier.dataset_id} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})")
        if self.__json_data is None:
            self.__json_data = self.__json_tooling.load_json_data()

        if dataset_identifier.dataset_id not in self.__json_data:
            # Handle dataset creation
            if create_if_not_exists:
                self.__json_data[dataset_identifier.dataset_id] = {}
            else:
                raise DatasetNotFoundException(dataset_identifier.dataset_id)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier.dataset_id} would be overridden")

        # Handle group creation
        if dataset_identifier.group_id not in self.__json_data[dataset_identifier.dataset_id]:
            if create_if_not_exists:
                self.__json_data[dataset_identifier.dataset_id][dataset_identifier.group_id] = {}
            else:
                raise DatasetNotFoundException(f'group {dataset_identifier.group_id} in dataset {dataset_identifier.dataset_id}')

        return self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write, data_types_dict=data_types_dict)

    def clear_dataset(self, dataset_id: str) -> None:
        """
        Utility method to remove the directory corresponding to a given dataset_id within the JSON database.

        Args:
            dataset_id (str): Identifier of the dataset to be removed
        """
        if self.__json_data is None:
            self.__json_data = self.__json_tooling.load_json_data()
        if dataset_id in self.__json_data:
            del self.__json_data[dataset_id]
        self.__json_tooling.save_json_data(self.__json_data)
