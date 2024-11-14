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

from sostrades_core.datasets.dataset_info.dataset_info_v1 import DatasetInfoV1
from sostrades_core.datasets.dataset_info.dataset_info_versions import VERSION_V1
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetGenericException,
    DatasetNotFoundException,
)
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import (
    DatasetSerializerType,
    DatasetsSerializerFactory,
)
from sostrades_core.tools.folder_operations import makedirs_safe, rmtree_safe


class LocalFileSystemDatasetsConnectorV1(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in local filesystem
    """
    DESCRIPTOR_FILE_NAME = 'descriptor.json'
    DATA_GROUP_DIRECTORY_KEY = '__data_group_filesystem_directory__'

    def __init__(self, connector_id: str, root_directory_path: str,
                 create_if_not_exists: bool = False,
                 serializer_type: DatasetSerializerType = DatasetSerializerType.FileSystem):
        """
        Constructor for Local Filesystem data connector

        Args:
            connector_id (str): Identifier for the connector.
            root_directory_path (str): Root directory path for this dataset connector using filesystem.
            create_if_not_exists (bool): Whether to create the root directory if it does not exist.
            serializer_type (DatasetSerializerType): Type of serializer to deserialize data from connector.
        """
        super().__init__()
        self._root_directory_path = os.path.abspath(root_directory_path)
        self._create_if_not_exists = create_if_not_exists

        # create dataset folder if it does not exists
        if self._create_if_not_exists and not os.path.isdir(self._root_directory_path):
            makedirs_safe(self._root_directory_path, exist_ok=True)

        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing local connector on {root_directory_path}")

        # configure dataset serializer
        self._datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)

        self.connector_id = connector_id
        self.compatible_dataset_info_version = [VERSION_V1]

    def __load_dataset_descriptor(self, dataset_id: str) -> dict[str, dict[str, Any]]:
        """
        Deal with all directories for the connector and read dataset_descriptor

        Args:
            dataset_id (str): Identifier of the dataset.

        Returns:
            dict[str, dict[str, Any]]: Dataset descriptor.
        """
        filesystem_dataset_identifier = self._datasets_serializer.format_filesystem_name(dataset_id)
        if filesystem_dataset_identifier != dataset_id:
            raise DatasetGenericException(f"Dataset {dataset_id} has a non-compliant name for connector {self}, "
                                          f"please use a compliant name such as  {filesystem_dataset_identifier} instead")

        dataset_directory = self.__build_dataset_path(dataset_id)
        dataset_descriptor_path = self.__build_descriptor_file_path(dataset_id)

        self._datasets_serializer.check_path_exists("Dataset folder", dataset_directory)

        # Load the descriptor, the serializer loads the pickle if it exists
        return self._datasets_serializer.read_descriptor_file(dataset_id, dataset_descriptor_path)

    def _get_values(self, dataset_identifier: DatasetInfoV1, data_to_get: dict[str, str]) -> dict[str, Any]:
        """
        Method to retrieve data from local dataset and fill a data_dict

        Args:
            dataset_identifier (DatasetInfoV1): Identifier of the dataset.
            data_to_get (dict[str, str]): Data to retrieve, dict of names and types.

        Returns:
            dict[str, Any]: Retrieved data.
        """
        self._logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier.dataset_id}, dataset group {dataset_identifier.group_id}, for connector {self}")

        dataset_descriptor = self.__load_dataset_descriptor(dataset_identifier.dataset_id)

        filesystem_data_group_id = self.__get_data_group_directory(dataset_descriptor, dataset_identifier.dataset_id, dataset_identifier.group_id)
        self._datasets_serializer.set_dataset_directory(self.__build_group_path(dataset_identifier.dataset_id,
                                                                      filesystem_data_group_id))

        self._datasets_serializer.load_pickle_data()

        # Filter data
        filtered_values = {key: self._datasets_serializer.convert_from_dataset_data(key,
                                                                                    self._extract_value_from_datum(dataset_descriptor[dataset_identifier.group_id][key]),
                                                                                    data_to_get)
                           for key in dataset_descriptor[dataset_identifier.group_id] if key in data_to_get}

        # Clear pickle buffer from serializer
        self._datasets_serializer.clear_pickle_data()

        self._logger.debug(f"Values obtained {list(filtered_values.keys())} for dataset {dataset_identifier.dataset_id} for connector {self}")
        return filtered_values

    def get_datasets_available(self) -> list[DatasetInfoV1]:
        """
        Get all available datasets for a specific API

        Returns:
            list[DatasetInfoV1]: List of datasets identifiers.
        """
        self._logger.debug(f"Getting all datasets for connector {self}")
        return [DatasetInfoV1(self.connector_id, dataset_id, group_id)
                for dataset_id in next(os.walk(self._root_directory_path))[1]
                for group_id in next(os.walk(self.__build_dataset_path(dataset_id)))[1]]

    def _write_values(self, dataset_identifier: DatasetInfoV1, values_to_write: dict[str, Any], data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Method to write data

        Args:
            dataset_identifier (DatasetInfoV1): Identifier of the dataset.
            values_to_write (dict[str, Any]): Dict of data to write {name: value}.
            data_types_dict (dict[str, str]): Dict of data type {name: type}.

        Returns:
            dict[str, Any]: Written values.
        """
        self._logger.debug(f"Writing values in dataset {dataset_identifier.dataset_id} for connector {self}")

        dataset_descriptor = self.__load_dataset_descriptor(dataset_identifier.dataset_id)

        filesystem_data_group_identifier = self._datasets_serializer.format_filesystem_name(dataset_identifier.group_id)
        data_group_dir = self.__build_group_path(dataset_identifier.dataset_id, filesystem_data_group_identifier)
        self._datasets_serializer.set_dataset_directory(data_group_dir)

        self._datasets_serializer.load_pickle_data()

        # Write data, serializer buffers the data to pickle and already pickled
        descriptor_values = {key: self._datasets_serializer.convert_to_dataset_data(key,
                                                                                    value,
                                                                                    data_types_dict)
                                for key, value in values_to_write.items()}

        dataset_descriptor[dataset_identifier.group_id] = dataset_descriptor.get(dataset_identifier.group_id, {})
        self._update_data_with_values(dataset_descriptor[dataset_identifier.group_id], descriptor_values, data_types_dict)

        dataset_descriptor = self.__index_data_group_directory(dataset_descriptor,
                                                               dataset_identifier.dataset_id,
                                                               dataset_identifier.group_id,
                                                               filesystem_data_group_identifier)

        # write in dataset descriptor
        dataset_descriptor_path = os.path.join(self._root_directory_path, dataset_identifier.dataset_id, self.DESCRIPTOR_FILE_NAME)
        self._datasets_serializer.write_descriptor_file(dataset_descriptor_path, dataset_descriptor)

        self._datasets_serializer.dump_pickle_data()
        return values_to_write

    def _get_values_all(self, dataset_identifier: DatasetInfoV1, data_types_dict: dict[str, str]) -> dict[str, Any]:
        """
        Abstract method to get all values from a dataset for a specific API

        Args:
            dataset_identifier (DatasetInfoV1): Identifier of the dataset.
            data_types_dict (dict[str, str]): Dict of data type {name: type}.

        Returns:
            dict[str, Any]: All values from the dataset.
        """
        dataset_descriptor = self.__load_dataset_descriptor(dataset_identifier.dataset_id)
        dataset_values = dict()
        filesystem_data_group_id = self.__get_data_group_directory(dataset_descriptor, dataset_identifier.dataset_id, dataset_identifier.group_id)
        self._datasets_serializer.set_dataset_directory(
                self.__build_group_path(dataset_identifier.dataset_id, filesystem_data_group_id))
        self._datasets_serializer.load_pickle_data()
        dataset_values = {
                key: self._datasets_serializer.convert_from_dataset_data(key,
                                                                          self._extract_value_from_datum(datum),
                                                                          data_types_dict)
                for key, datum in dataset_descriptor[dataset_identifier.group_id].items()
                if datum != filesystem_data_group_id}
        self._datasets_serializer.clear_pickle_data()
        return dataset_values

    def _write_dataset(self, dataset_identifier: DatasetInfoV1, values_to_write: dict[str, Any], data_types_dict: dict[str, str], create_if_not_exists: bool = True, override: bool = False) -> None:
        """
        Abstract method to overload in order to write a dataset from a specific API

        Args:
            dataset_identifier (DatasetInfoV1): Identifier of the dataset.
            values_to_write (dict[str, Any]): Dict of data to write {name: value}.
            data_types_dict (dict[str, str]): Dict of data types {name: type}.
            create_if_not_exists (bool): Create the dataset if it does not exists (raises otherwise).
            override (bool): Override dataset if it exists (raises otherwise).
        """
        self._logger.debug(f"Writing dataset {dataset_identifier.dataset_id} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})")

        dataset_directory = self.__build_dataset_path(dataset_identifier.dataset_id)
        dataset_descriptor_path = self.__build_descriptor_file_path(dataset_identifier.dataset_id)

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

    def clear(self, remove_root_directory: bool = False) -> None:
        """
        Utility method to remove all datasets in the connector root directory.

        Args:
            remove_root_directory (bool): Whether to delete the root directory itself too.
        """
        if remove_root_directory:
            rmtree_safe(self._root_directory_path)
        else:
            map(self.clear_dataset, self.get_datasets_available())

    def clear_dataset(self, dataset_id: str) -> None:
        """
        Utility method to remove the directory corresponding to a given dataset_id within the root directory.

        Args:
            dataset_id (str): Identifier of the dataset to be removed.
        """
        rmtree_safe(os.path.join(self._root_directory_path, dataset_id))

    def __index_data_group_directory(self,
                                     dataset_descriptor: dict[str, dict[str, Any]],
                                     dataset_identifier: str,
                                     data_group_identifier: str,
                                     filesystem_data_group_identifier: str) -> dict[str, dict[str, Any]]:
        """
        Index the data group directory.

        Args:
            dataset_descriptor (dict[str, dict[str, Any]]): Dataset descriptor.
            dataset_identifier (str): Identifier of the dataset.
            data_group_identifier (str): Identifier of the data group.
            filesystem_data_group_identifier (str): Filesystem identifier of the data group.

        Returns:
            dict[str, dict[str, Any]]: Updated dataset descriptor.
        """
        if filesystem_data_group_identifier != data_group_identifier:
            dataset_descriptor[data_group_identifier][self.DATA_GROUP_DIRECTORY_KEY] = filesystem_data_group_identifier
            existing_group_dirs = {dataset_descriptor[_group_id].get(self.DATA_GROUP_DIRECTORY_KEY, _group_id): _group_id
                                   for _group_id in dataset_descriptor if _group_id != data_group_identifier}
            if filesystem_data_group_identifier in existing_group_dirs:
                self.__logger.error(f"Dataset {dataset_identifier} of connector {self} is storing "
                                    f"{existing_group_dirs[filesystem_data_group_identifier]} and "
                                    f"{data_group_identifier} in the same directory {filesystem_data_group_identifier},"
                                    f" please change data group names to avoid conflicting storage.")
        return dataset_descriptor

    def __get_data_group_directory(self,
                                   dataset_descriptor: dict[str, dict[str, Any]],
                                   dataset_identifier: str,
                                   data_group_identifier: str) -> str:
        """
        Get the data group directory.

        Args:
            dataset_descriptor (dict[str, dict[str, Any]]): Dataset descriptor.
            dataset_identifier (str): Identifier of the dataset.
            data_group_identifier (str): Identifier of the data group.

        Returns:
            str: Data group directory.
        """
        data_group_dir = dataset_descriptor[data_group_identifier].get(self.DATA_GROUP_DIRECTORY_KEY, data_group_identifier)
        formatted_data_group_dir = self._datasets_serializer.format_filesystem_name(data_group_dir)
        if data_group_dir != formatted_data_group_dir:
            self.__logger.error(f"Dataset {dataset_identifier} of connector {self} defines non-compliant "
                                f"directory name {data_group_dir} for the storage of the data group "
                                f"{data_group_identifier}, using compliant {formatted_data_group_dir} instead")
        return formatted_data_group_dir

    def __build_descriptor_file_path(self, dataset_id: str) -> str:
        """
        Build the dataset descriptor file path that is in the dataset.

        Args:
            dataset_id (str): Identifier of the dataset.

        Returns:
            str: Dataset descriptor file path.
        """
        return os.path.join(self.__build_dataset_path(dataset_id), self.DESCRIPTOR_FILE_NAME)

    def __build_dataset_path(self, dataset_id: str) -> str:
        """
        Build the dataset folder path that is in the root folder of the connector.

        Args:
            dataset_id (str): Identifier of the dataset.

        Returns:
            str: Dataset folder path.
        """
        return os.path.join(self._root_directory_path, dataset_id)

    def __build_group_path(self, dataset_id: str, group_name: str) -> str:
        """
        Build the dataset descriptor file path that is in the dataset.

        Args:
            dataset_id (str): Identifier of the dataset.
            group_name (str): Name of the group.

        Returns:
            str: Group path.
        """
        return os.path.join(self._root_directory_path, dataset_id, group_name)
