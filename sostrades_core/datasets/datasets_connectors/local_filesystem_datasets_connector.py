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
    DatasetDeserializeException,
    DatasetGenericException,
    DatasetNotFoundException,
)
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import (
    DatasetSerializerType,
    DatasetsSerializerFactory,
)
from sostrades_core.tools.folder_operations import makedirs_safe, rmtree_safe


class LocalFileSystemDatasetsConnector(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in local filesystem
    """
    DESCRIPTOR_FILE_NAME = 'descriptor.json'
    DATA_GROUP_DIRECTORY_KEY = '__data_group_filesystem_directory__'


    def __init__(self, root_directory_path: str,
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
        self.__root_directory_path = os.path.abspath(root_directory_path)
        self._create_if_not_exists = create_if_not_exists
        if self._create_if_not_exists and not os.path.isdir(self.__root_directory_path):
            makedirs_safe(self.__root_directory_path, exist_ok=True)
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug(f"Initializing local connector on {root_directory_path}")
        self.__datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)

    def __format_filesystem_name(self, fs_name):
        """
        Uses serializer capability to format a filesystem name so that it is compatible with Windows, Ubuntu and MacOS.
        Used at connector level to format data group identifiers and to check the compliance of dataset identifiers.
        :param fs_name: filesystem name as defined by the user or namespace in case of wildcards.
        :return: formatted filesystem name filesystem-compatible using utf-8 encoding of forbidden characters.
        """
        return self.__datasets_serializer.format_filesystem_name(fs_name)

    def __load_dataset_descriptor(self, dataset_identifier: str) -> dict[str: dict[str: Any]]:
        """
        Method to load dataset descriptor from JSON file containing the basic types variables as well as the dataset
        descriptor values type "@dataframe@d.csv" for the types stored in filesystem.
        :param dataset_identifier: identifier of the dataset whose descriptor is to be loaded
        :type dataset_identifier: str
        :return: dictionary {data_group: {parameter_name: parameter_data_and_metadata}}
        """
        if not os.path.exists(self.__root_directory_path):
            raise DatasetGenericException(f"Datasets database folder not found at {self.__root_directory_path}.")
        filesystem_dataset_identifier = self.__format_filesystem_name(dataset_identifier)
        if filesystem_dataset_identifier != dataset_identifier:
            raise DatasetGenericException(f"Dataset {dataset_identifier} has a non-compliant name for connector {self}, "
                                          f"please use a compliant name such as  {filesystem_dataset_identifier} instead")
        dataset_directory = os.path.join(self.__root_directory_path, dataset_identifier)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)

        if not os.path.exists(dataset_descriptor_path):
            raise DatasetNotFoundException(dataset_identifier)
        descriptor_data = None
        try:
            with open(dataset_descriptor_path, "r", encoding="utf-8") as file:
                descriptor_data = json.load(fp=file)
        except TypeError as exception:
            raise DatasetDeserializeException(dataset_identifier, f'type error exception in dataset descriptor, {str(exception)}')
        except json.JSONDecodeError as exception:
            raise DatasetDeserializeException(dataset_identifier, f'dataset descriptor does not have a valid json format, {str(exception.msg)}')
        return descriptor_data

    def __save_dataset_descriptor_and_pickle(self, dataset_identifier: str, descriptor_data: dict[str: Any]) -> None:
        """
        Method to save dataset descriptor into JSON file containing the basic types variables as well as the dataset
        descriptor values type "@dataframe@d.csv" for the types stored in filesystem.
        :param dataset_identifier: identifier of the dataset whose descriptor is to be saved
        :type dataset_identifier: str
        :return: dictionary of descriptor keys and values
        :param descriptor_data: data as it will be saved into te descriptor
        :type descriptor_data: dict
        """
        dataset_directory = os.path.join(self.__root_directory_path, dataset_identifier)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)
        if not os.path.exists(dataset_descriptor_path):
            raise DatasetGenericException() from FileNotFoundError(f"The dataset descriptor json file is not found at "
                                                                   f"{dataset_descriptor_path}")
        with open(dataset_descriptor_path, "w", encoding="utf-8") as file:
            json.dump(obj=descriptor_data, fp=file, indent=4)
        self.__datasets_serializer.dump_pickle_data()

    def __load_pickle_data(self):
        self.__datasets_serializer.load_pickle_data()

    def __clear_pickle_data(self):
        self.__datasets_serializer.clear_pickle_data()

    def get_values(self, dataset_identifier: str, data_group_identifier: str, data_to_get: dict[str:str]) -> dict[str:Any]:
        """
        Method to retrieve data from a single data group within the local dataset and fill a data_dict
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

        # Load the descriptor, the serializer loads the pickle if it exists
        dataset_descriptor = self.__load_dataset_descriptor(dataset_identifier=dataset_identifier)

        filesystem_data_group_id = self.__get_data_group_directory(dataset_descriptor, dataset_identifier, data_group_identifier)
        self.__datasets_serializer.set_dataset_directory(os.path.join(self.__root_directory_path,
                                                                      dataset_identifier,
                                                                      filesystem_data_group_id))
        self.__load_pickle_data()
        # Filter data
        filtered_values = {key: self.__datasets_serializer.convert_from_dataset_data(key,
                                                                                     self._extract_value_from_datum(dataset_descriptor[data_group_identifier][key]),
                                                                                     data_to_get)
                           for key in dataset_descriptor[data_group_identifier] if key in data_to_get}
        # Clear pickle buffer from serializer
        self.__clear_pickle_data()
        self.__logger.debug(f"Values obtained {list(filtered_values.keys())} for dataset {dataset_identifier} for connector {self}")
        return filtered_values

    def get_datasets_available(self) -> list[str]:
        """
        Get all available datasets for a specific API
        :return: list of datasets identifiers
        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        return next(os.walk(self.__root_directory_path))[1]

    def write_values(self, dataset_identifier: str, data_group_identifier: str, values_to_write: dict[str:Any],
                     data_types_dict: dict[str:str]) -> dict[str:Any]:
        """
        Method to write data
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_group_identifier: identifier of the data group inside the dataset
        :type data_group_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str], name, value
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        :return: values_to_write
        """
        self.__logger.debug(f"Writing values in dataset {dataset_identifier} for connector {self}")
        # read the already existing values
        dataset_descriptor = self.__load_dataset_descriptor(dataset_identifier=dataset_identifier)

        filesystem_data_group_identifier = self.__format_filesystem_name(data_group_identifier)
        data_group_dir = os.path.join(self.__root_directory_path, dataset_identifier, filesystem_data_group_identifier)
        self.__datasets_serializer.set_dataset_directory(data_group_dir)

        self.__load_pickle_data()
        # Write data, serializer buffers the data to pickle and already pickled
        descriptor_values = {key: self.__datasets_serializer.convert_to_dataset_data(key,
                                                                                     value,
                                                                                     data_types_dict)
                             for key, value in values_to_write.items()}
        if data_group_identifier not in dataset_descriptor:
            dataset_descriptor[data_group_identifier] = dict()
        self._update_data_with_values(dataset_descriptor[data_group_identifier],
                                      descriptor_values, data_types_dict)
        dataset_descriptor = self.__index_data_group_directory(dataset_descriptor,
                                                               dataset_identifier,
                                                               data_group_identifier,
                                                               filesystem_data_group_identifier)

        self.__save_dataset_descriptor_and_pickle(dataset_identifier=dataset_identifier,
                                                  descriptor_data=dataset_descriptor)
        return values_to_write

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
        formatted_data_group_dir = self.__format_filesystem_name(data_group_dir)
        if data_group_dir != formatted_data_group_dir:
            self.__logger.error(f"Dataset {dataset_identifier} of connector {self} defines non-compliant "
                                f"directory name {data_group_dir} for the storage of the data group "
                                f"{data_group_identifier}, using compliant {formatted_data_group_dir} instead")
        return formatted_data_group_dir

    def get_values_all(self, dataset_identifier: str, data_types_dict: dict[str:dict[str:str]]) -> dict[str:dict[str:Any]]:
        """
        Abstract method to get all values from a dataset for the local filesystem connector.
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_types_dict: dict of data type by data group {data_group: {name: type}}
        :type data_types_dict: dict[str:dict[str:str]]
        :return: dataset values by group
        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier} for connector {self}")

        dataset_descriptor = self.__load_dataset_descriptor(dataset_identifier=dataset_identifier)
        dataset_values_by_group = dict()
        for _group_id, _group_data in dataset_descriptor.items():
            filesystem_data_group_id = self.__get_data_group_directory(dataset_descriptor, dataset_identifier, _group_id)
            self.__datasets_serializer.set_dataset_directory(
                os.path.join(self.__root_directory_path, dataset_identifier, filesystem_data_group_id))
            self.__load_pickle_data()
            dataset_values_by_group[_group_id] = {
                key: self.__datasets_serializer.convert_from_dataset_data(key,
                                                                          self._extract_value_from_datum(datum),
                                                                          data_types_dict[_group_id])
                for key, datum in _group_data.items()}
            self.__clear_pickle_data()
        return dataset_values_by_group

    def write_dataset(self, dataset_identifier: str, values_to_write: dict[str:dict[str:Any]],
                      data_types_dict: dict[str:dict[str:str]], create_if_not_exists: bool = True, override: bool = False
                      ) -> dict[str:dict[str:Any]]:
        """
        Abstract method to overload in order to write a dataset for the local filesystem connector.
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
        filesystem_dataset_identifier = self.__format_filesystem_name(dataset_identifier)
        if filesystem_dataset_identifier != dataset_identifier:
            raise DatasetGenericException(f"Dataset {dataset_identifier} has a non-compliant name for connector {self}, "
                                          f"please use a compliant name such as  {filesystem_dataset_identifier} instead")
        dataset_directory = os.path.join(self.__root_directory_path, dataset_identifier)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)
        if not os.path.exists(dataset_descriptor_path):
            # Handle dataset creation
            if create_if_not_exists:
                makedirs_safe(dataset_directory, exist_ok=True)
                with open(dataset_descriptor_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
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

    def clear(self, remove_root_directory:bool=False) -> None:
        """
        Utility method to remove all datasets in the connector root directory.
        :param remove_root_directory: whether to delete the root directory itself too.
        :type remove_root_directory: bool
        :return: None
        """
        if remove_root_directory:
            rmtree_safe(self.__root_directory_path)
        else:
            map(self.clear_dataset, self.get_datasets_available())

    def clear_dataset(self, dataset_id: str) -> None:
        """
        Utility method to remove the directory corresponding to a given dataset_id within the root directory.
        :param dataset_id: identifier of the dataset to be removed
        :type dataset_id: str
        :return: None
        """
        rmtree_safe(os.path.join(self.__root_directory_path, self.__format_filesystem_name(dataset_id)))
