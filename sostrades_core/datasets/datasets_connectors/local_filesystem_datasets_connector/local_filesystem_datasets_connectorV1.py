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


        :param root_directory_path: root directory path for this dataset connector using filesystem
        :type root_directory_path: str
        :param create_if_not_exists: whether to create the root directory if it does not exist
        :type create_if_not_exists: bool
        :param serializer_type: type of serializer to deserialize data from connector
        :type serializer_type: DatasetSerializerType (JSON for jsonDatasetSerializer)
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

    def __load_dataset_descriptor(self, dataset_id: str) -> dict[str: dict[str: Any]]:
        """
        Deal with all directories for the connector and read dataset_descriptor
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


    def _get_values(self, dataset_identifier: DatasetInfoV1, data_to_get: dict[str:str]) -> dict[str:Any]:
        """
        Method to retrieve data from local dataset and fill a data_dict

        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: DatasetInfo

        :param data_to_get: data to retrieve, dict of names and types
        :type data_to_get: dict[str:str]
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
        :return: list of datasets identifiers
        """
        self._logger.debug(f"Getting all datasets for connector {self}")
        return [DatasetInfoV1(self.connector_id, dataset_id, group_id)
                for dataset_id in next(os.walk(self._root_directory_path))[1]
                for group_id in next(os.walk(self.__build_dataset_path(dataset_id)))[1]]

    def _write_values(self, dataset_identifier: DatasetInfoV1, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> dict[str: Any]:
        """
        Method to write data
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str], name, value
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        :return: None
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

    def _get_values_all(self, dataset_identifier: DatasetInfoV1, data_types_dict: dict[str:str]) -> dict[str:Any]:
        """
        Abstract method to get all values from a dataset for a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: DatasetInfo
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        :return: None
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

    def _write_dataset(self, dataset_identifier: DatasetInfoV1, values_to_write: dict[str:Any], data_types_dict: dict[str:str], create_if_not_exists: bool = True, override: bool = False) -> None:
        """
        Abstract method to overload in order to write a dataset from a specific API
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
        :return: None
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
        :param remove_root_directory: whether to delete the root directory itself too.
        :type remove_root_directory: bool
        :return: None
        """
        if remove_root_directory:
            rmtree_safe(self._root_directory_path)
        else:
            map(self.clear_dataset, self.get_datasets_available())

    def clear_dataset(self, dataset_id: str) -> None:
        """
        Utility method to remove the directory corresponding to a given dataset_id within the root directory.
        :param dataset_id: identifier of the dataset to be removed
        :type dataset_id: str
        :return: None
        """
        rmtree_safe(os.path.join(self._root_directory_path, dataset_id))


    def __index_data_group_directory(self,
                                     dataset_descriptor,
                                     dataset_identifier,
                                     data_group_identifier,
                                     filesystem_data_group_identifier):
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
                                   dataset_descriptor,
                                   dataset_identifier,
                                   data_group_identifier):
        data_group_dir = dataset_descriptor[data_group_identifier].get(self.DATA_GROUP_DIRECTORY_KEY, data_group_identifier)
        formatted_data_group_dir = self._datasets_serializer.format_filesystem_name(data_group_dir)
        if data_group_dir != formatted_data_group_dir:
            self.__logger.error(f"Dataset {dataset_identifier} of connector {self} defines non-compliant "
                                f"directory name {data_group_dir} for the storage of the data group "
                                f"{data_group_identifier}, using compliant {formatted_data_group_dir} instead")
        return formatted_data_group_dir

    def __build_descriptor_file_path(self, dataset_id:str)->str:
        """
        build the dataset descriptor file path that is in the dataset
        """
        return os.path.join(self.__build_dataset_path(dataset_id), self.DESCRIPTOR_FILE_NAME)

    def __build_dataset_path(self, dataset_id:str)->str:
        """
        build the dataset foler path that is in the root folder of the connector
        """
        return os.path.join(self._root_directory_path, dataset_id)

    def __build_group_path(self, dataset_id:str, group_name:str)->str:
        """
        build the dataset descriptor file path that is in the dataset
        """
        return os.path.join(self._root_directory_path, dataset_id, group_name)
