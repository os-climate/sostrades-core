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

import json
import logging
import os
from typing import Any

from sostrades_core.datasets.dataset_info.dataset_info_v0 import DatasetInfoV0
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    DatasetGenericException,
    DatasetNotFoundException,
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.\
    local_filesystem_datasets_connector_base import LocalFileSystemDatasetsConnectorBase
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import (
    DatasetSerializerType,
)
from sostrades_core.tools.folder_operations import makedirs_safe


class LocalFileSystemDatasetsConnectorV0(LocalFileSystemDatasetsConnectorBase):
    """Specific dataset connector for dataset in local filesystem"""

    DESCRIPTOR_FILE_NAME = 'descriptor.json'

    def __init__(self, connector_id: str, root_directory_path: str,
                 create_if_not_exists: bool = False,
                 serializer_type: DatasetSerializerType = DatasetSerializerType.FileSystem):
        """
        Constructor for Local Filesystem data connector

        Args:
            connector_id (str): The identifier for the connector.
            root_directory_path (str): Root directory path for this dataset connector using filesystem.
            create_if_not_exists (bool, optional): Whether to create the root directory if it does not exist. Defaults to False.
            serializer_type (DatasetSerializerType, optional): Type of serializer to deserialize data from connector. Defaults to DatasetSerializerType.FileSystem.

        """
        super().__init__(connector_id=connector_id,
                         root_directory_path=root_directory_path,
                         create_if_not_exists=create_if_not_exists,
                         serializer_type=serializer_type)
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing local connector V0 on {root_directory_path}")

    def _get_values(self, dataset_identifier: DatasetInfoV0, data_to_get: dict[str, str]) -> dict[str, Any]:
        """
        Method to retrieve data from local dataset and fill a data_dict

        Args:
            dataset_identifier (DatasetInfoV0): Identifier of the dataset.
            data_to_get (dict[str, str]): Data to retrieve, dict of names and types.

        Returns:
            dict[str, Any]: Retrieved data.

        """
        self._logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier.dataset_id} for connector {self}")

        filesystem_dataset_identifier = self._datasets_serializer.format_filesystem_name(dataset_identifier.dataset_id)
        if filesystem_dataset_identifier != dataset_identifier.dataset_id:
            raise DatasetGenericException(f"Dataset {dataset_identifier.dataset_id} has a non-compliant name for connector {self}, "
                                          f"please use a compliant name such as  {filesystem_dataset_identifier} instead")

        dataset_directory = os.path.join(self._root_directory_path, dataset_identifier.dataset_id)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)

        self._datasets_serializer.check_path_exists(self.DATASET_FOLDER, dataset_directory)

        self._datasets_serializer.set_dataset_directory(dataset_directory)

        # Load the descriptor, the serializer loads the pickle if it exists
        dataset_descriptor = self._datasets_serializer.read_descriptor_file(dataset_identifier.dataset_id, dataset_descriptor_path)
        self._datasets_serializer.load_pickle_data()

        # Filter data
        filtered_values = {key: self._datasets_serializer.convert_from_dataset_data(key,
                                                                                    self._extract_value_from_datum(dataset_descriptor[key]),
                                                                                    data_to_get)
                           for key in dataset_descriptor if key in data_to_get}

        # Clear pickle buffer from serializer
        self._datasets_serializer.clear_pickle_data()

        self._logger.debug(f"Values obtained {list(filtered_values.keys())} for dataset {dataset_identifier.dataset_id} for connector {self}")
        return filtered_values

    def get_datasets_available(self) -> list[DatasetInfoV0]:
        """
        Get all available datasets for a specific API

        Returns:
            list[DatasetInfoV0]: List of datasets identifiers.

        """
        self._logger.debug(f"Getting all datasets for connector {self}")
        return [DatasetInfoV0(self.connector_id, dataset_id) for dataset_id in next(os.walk(self._root_directory_path))[1]]

    def _write_values(self, dataset_identifier: DatasetInfoV0, values_to_write: dict[str, Any], data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Method to write data

        Args:
            dataset_identifier (DatasetInfoV0): Dataset identifier for connector.
            values_to_write (dict[str, Any]): Dict of data to write {name: value}.
            data_types_dict (dict[str, str]): Dict of data type {name: type}.

        Returns:
            dict[str, Any]: Written values.

        """
        self._logger.debug(f"Writing values in dataset {dataset_identifier.dataset_id} for connector {self}")

        dataset_directory = os.path.join(self._root_directory_path, dataset_identifier.dataset_id)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)

        self._datasets_serializer.check_path_exists(self.DATASET_FOLDER, dataset_directory)

        self._datasets_serializer.set_dataset_directory(dataset_directory)

        # read the already existing values
        dataset_descriptor = self._datasets_serializer.read_descriptor_file(dataset_identifier.dataset_id, dataset_descriptor_path)

        # Write data, serializer buffers the data to pickle and already pickled
        descriptor_values = {key: self._datasets_serializer.convert_to_dataset_data(key,
                                                                                    value,
                                                                                    data_types_dict)
                             for key, value in values_to_write.items()}

        self._update_data_with_values(dataset_descriptor, descriptor_values, data_types_dict)

        # write in dataset descriptor
        self._datasets_serializer.write_descriptor_file(dataset_descriptor_path, dataset_descriptor)

        self._datasets_serializer.dump_pickle_data()
        return values_to_write

    def _get_values_all(self, dataset_identifier: DatasetInfoV0, data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Abstract method to get all values from a dataset for a specific API

        Args:
            dataset_identifier (DatasetInfoV0): Dataset identifier for connector.
            data_types_dict (dict[str, str]): Dict of data type {name: type}.

        Returns:
            dict[str, Any]: All values from the dataset.

        """
        self._logger.debug(f"Getting all values for dataset {dataset_identifier.dataset_id} for connector {self}")
        dataset_directory = os.path.join(self._root_directory_path, dataset_identifier.dataset_id)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)

        self._datasets_serializer.check_path_exists(self.DATASET_FOLDER, dataset_directory)

        self._datasets_serializer.set_dataset_directory(dataset_directory)

        # Load the descriptor, the serializer loads the pickle if it exists
        dataset_descriptor = self._datasets_serializer.read_descriptor_file(dataset_identifier.dataset_id, dataset_descriptor_path)

        self._datasets_serializer.load_pickle_data()

        # Filter data
        filtered_values = {key: self._datasets_serializer.convert_from_dataset_data(key,
                                                                                    self._extract_value_from_datum(dataset_descriptor[key]),
                                                                                    data_types_dict)
                           for key in dataset_descriptor}

        # Clear pickle buffer from serializer
        self._datasets_serializer.clear_pickle_data()
        return filtered_values

    def _write_dataset(self, dataset_identifier: DatasetInfoV0, values_to_write: dict[str, Any], data_types_dict: dict[str, str], create_if_not_exists: bool = True, override: bool = False) -> dict[str, Any]:
        """
        Abstract method to overload in order to write a dataset from a specific API

        Args:
            dataset_identifier (DatasetInfoV0): Dataset identifier for connector.
            values_to_write (dict[str, Any]): Dict of data to write {name: value}.
            data_types_dict (dict[str, str]): Dict of data types {name: type}.
            create_if_not_exists (bool, optional): Create the dataset if it does not exists (raises otherwise). Defaults to True.
            override (bool, optional): Override dataset if it exists (raises otherwise). Defaults to False.

        """
        self._logger.debug(f"Writing dataset {dataset_identifier.dataset_id} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})")

        dataset_directory = os.path.join(self._root_directory_path, dataset_identifier.dataset_id)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)

        if not os.path.exists(dataset_descriptor_path):
            # Handle dataset creation
            if create_if_not_exists:
                makedirs_safe(dataset_directory, exist_ok=True)
                with open(dataset_descriptor_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            else:
                raise DatasetNotFoundException(dataset_identifier.dataset_id)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier.dataset_id} would be overriden")

        return self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write, data_types_dict=data_types_dict)
