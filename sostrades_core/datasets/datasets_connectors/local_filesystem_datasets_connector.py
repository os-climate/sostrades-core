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
import json, pickle
import logging
import os
from shutil import rmtree
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


class LocalFileSystemDatasetsConnector(AbstractDatasetsConnector):
    """
    Specific dataset connector for dataset in local filesystem
    """
    DESCRIPTOR_FILE_NAME = 'descriptor.json'
    NON_SERIALIZABLE_PKL = 'non_serializable.pkl'

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
            os.makedirs(self.__root_directory_path, exist_ok=True)
        self.__logger = logging.getLogger(__name__)
        self.__logger.debug(f"Initializing local connector on {root_directory_path}")
        self.__datasets_serializer = DatasetsSerializerFactory.get_serializer(serializer_type)

    def __load_dataset_descriptor_and_pickle(self, dataset_identifier: str) -> dict[str: Any]:
        """
        Method to load dataset descriptor from JSON file containing the basic types variables as well as the dataset
        descriptor values type "@dataframe@d.csv" for the types stored in filesystem.
        :param dataset_identifier: identifier of the dataset whose descriptor is to be loaded
        :type dataset_identifier: str
        :return: dictionary of descriptor keys and values
        """
        if not os.path.exists(self.__root_directory_path):
            raise DatasetGenericException(f"Datasets database folder not found at {self.__root_directory_path}.")
        
        dataset_directory = os.path.join(self.__root_directory_path, dataset_identifier)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)
        non_serializable_pkl_path = os.path.join(dataset_directory, self.NON_SERIALIZABLE_PKL)
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

        if os.path.exists(non_serializable_pkl_path):
            pkl_data = None
            try:
                with open(non_serializable_pkl_path, 'rb') as pkl_file:
                    pkl_data = pickle.load(pkl_file)
            except TypeError as exception:
                raise DatasetDeserializeException(dataset_identifier,
                                                  f'type error exception in dataset pickle file for non-serializable data, {str(exception)}')
            except pickle.UnpicklingError as exception:
                raise DatasetDeserializeException(dataset_identifier,
                                                  f'dataset pickle file for non-serializable data does not have a valid pickle format, {str(exception)}')
        else:
            pkl_data = {}
        return descriptor_data, pkl_data

    def __save_dataset_descriptor_and_pickle(self, dataset_identifier: str, descriptor_data: dict[str: Any], data_to_pickle: dict[str: Any]) -> None:
        """
        Method to save dataset descriptor into JSON file containing the basic types variables as well as the dataset
        descriptor values type "@dataframe@d.csv" for the types stored in filesystem.
        :param dataset_identifier: identifier of the dataset whose descriptor is to be saved
        :type dataset_identifier: str
        :return: dictionary of descriptor keys and values
        :param descriptor_data: data as it will be saved into te descriptor
        :type descriptor_data: dict
        """
        # TODO: exception mgmt?
        dataset_directory = os.path.join(self.__root_directory_path, dataset_identifier)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)
        if not os.path.exists(dataset_descriptor_path):
            raise DatasetGenericException() from FileNotFoundError(f"The dataset descriptor json file is not found at "
                                                                   f"{dataset_descriptor_path}")
        with open(dataset_descriptor_path, "w", encoding="utf-8") as file:
            json.dump(obj=descriptor_data, fp=file, indent=4)

        if data_to_pickle:
            non_serializable_pkl_path = os.path.join(dataset_directory, self.NON_SERIALIZABLE_PKL)
            with open(non_serializable_pkl_path, 'wb') as pkl_file:
                pickle.dump(data_to_pickle, pkl_file)

    def get_values(self, dataset_identifier: str, data_to_get: dict[str:str]) -> dict[str:Any]:
        """
        Method to retrieve data from local dataset and fill a data_dict

        :param dataset_identifier: identifier of the dataset
        :type dataset_identifier: str

        :param data_to_get: data to retrieve, dict of names and types
        :type data_to_get: dict[str:str]
        """
        self.__logger.debug(f"Getting values {data_to_get.keys()} for dataset {dataset_identifier} for connector {self}")

        dataset_descriptor, pickled_data = self.__load_dataset_descriptor_and_pickle(dataset_identifier=dataset_identifier)
        self.__datasets_serializer.set_dataset_directory(os.path.join(self.__root_directory_path, dataset_identifier))

        # Filter data
        filtered_data = {key: self.__datasets_serializer.convert_from_dataset_data(key,
                                                                                   dataset_descriptor[key],
                                                                                   data_to_get)
                        for key in dataset_descriptor if key in data_to_get}

        # update with pickled data what the serializer pre-filled with @object@. KeyError if mismatch
        filtered_data.update({key: pickled_data[key] for key in filtered_data if
                              filtered_data[key] == self.__datasets_serializer.TYPE_OBJECT})
        self.__logger.debug(f"Values obtained {list(filtered_data.keys())} for dataset {dataset_identifier} for connector {self}")
        return filtered_data

    def get_datasets_available(self) -> list[str]:
        """
        Get all available datasets for a specific API
        :return: list of datasets identifiers
        """
        self.__logger.debug(f"Getting all datasets for connector {self}")
        return next(os.walk(self.__root_directory_path))[1]

    def write_values(self, dataset_identifier: str, values_to_write: dict[str:Any], data_types_dict: dict[str:str]) -> None:
        """
        Method to write data
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param values_to_write: dict of data to write {name: value}
        :type values_to_write: dict[str], name, value
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        :return: None
        """
        self.__logger.debug(f"Writing values in dataset {dataset_identifier} for connector {self}")
        # read the already existing values
        dataset_descriptor, pickled_data = self.__load_dataset_descriptor_and_pickle(dataset_identifier=dataset_identifier)
        self.__datasets_serializer.set_dataset_directory(os.path.join(self.__root_directory_path, dataset_identifier))

        # Write data
        dataset_descriptor.update({key: self.__datasets_serializer.convert_to_dataset_data(key,
                                                                                           value,
                                                                                           data_types_dict)
                                   for key, value in values_to_write.items()})

        # dataset descriptor contains potentially non-jsonifiable values
        dataset_descriptor, pickled_data = self.__update_pickle_data(dataset_descriptor, pickled_data, dataset_identifier)

        self.__save_dataset_descriptor_and_pickle(dataset_identifier=dataset_identifier,
                                                  descriptor_data=dataset_descriptor,
                                                  data_to_pickle=pickled_data)

    def __update_pickle_data(self, dataset_descriptor, pickled_data, dataset_id):
        # the dataset_descriptor contains the ground truth of what needs to be saved but might not be jsonifiable
        for key, value in dataset_descriptor.items():
            datum_dict = {key: value}
            try:
                _ = json.dumps(datum_dict)  # TODO: might be optimized by saving all the json strings
                # if no error then datum needs to be deleted from the pickled data to avoid spurious overwrite
                if key in pickled_data:
                    del pickled_data[key]
            except TypeError:  # non-jsonifiable
                self.__logger.debug(f"For {dataset_id}, parameter {key} is stored in pickle")
                pickled_data.update(datum_dict)
                dataset_descriptor[key] = self.__datasets_serializer.TYPE_OBJECT
        return dataset_descriptor, pickled_data

    def get_values_all(self, dataset_identifier: str, data_types_dict: dict[str:str]) -> dict[str:Any]:
        """
        Abstract method to get all values from a dataset for a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
        :param data_types_dict: dict of data type {name: type}
        :type data_types_dict: dict[str:str]
        :return: None
        """
        self.__logger.debug(f"Getting all values for dataset {dataset_identifier} for connector {self}")
        dataset_descriptor, pickled_data = self.__load_dataset_descriptor_and_pickle(dataset_identifier=dataset_identifier)
        self.__datasets_serializer.set_dataset_directory(os.path.join(self.__root_directory_path, dataset_identifier))

        dataset_data = {key: self.__datasets_serializer.convert_from_dataset_data(key,
                                                                                 value,
                                                                                 data_types_dict)
                        for key, value in dataset_descriptor.items()}

        # update with pickled data what the serializer pre-filled with @object@. KeyErrror if mismatch
        dataset_data.update({key: pickled_data[key] for key in dataset_data
                             if dataset_data[key] == self.__datasets_serializer.TYPE_OBJECT})
        return dataset_data

    def write_dataset(self, dataset_identifier: str, values_to_write: dict[str:Any], data_types_dict:dict[str:str], create_if_not_exists:bool=True, override:bool=False) -> None:
        """
        Abstract method to overload in order to write a dataset from a specific API
        :param dataset_identifier: dataset identifier for connector
        :type dataset_identifier: str
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
        self.__logger.debug(f"Writing dataset {dataset_identifier} for connector {self} (override={override}, create_if_not_exists={create_if_not_exists})")
        dataset_directory = os.path.join(self.__root_directory_path, dataset_identifier)
        dataset_descriptor_path = os.path.join(dataset_directory, self.DESCRIPTOR_FILE_NAME)

        if not os.path.exists(dataset_descriptor_path):
            # Handle dataset creation
            if create_if_not_exists:
                os.makedirs(dataset_directory, exist_ok=True)
                with open(dataset_descriptor_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            else:
                raise DatasetNotFoundException(dataset_identifier)
        else:
            # Handle override
            if not override:
                raise DatasetGenericException(f"Dataset {dataset_identifier} would be overriden")
        self.write_values(dataset_identifier=dataset_identifier, values_to_write=values_to_write, data_types_dict=data_types_dict)

    def clear(self, remove_root_directory:bool=False) -> None:
        """
        Utility method to remove all datasets in the connector root directory.
        :param remove_root_directory: whether to delete the root directory itself too.
        :type remove_root_directory: bool
        :return: None
        """
        if remove_root_directory:
            rmtree(self.__root_directory_path)
        else:
            map(self.clear_dataset, self.get_datasets_available())

    def clear_dataset(self, dataset_id: str) -> None:
        """
        Utility method to remove the directory corresponding to a given dataset_id within the root directory.
        :param dataset_id: identifier of the dataset to be removed
        :type dataset_id: str
        :return: None
        """
        rmtree(os.path.join(self.__root_directory_path, dataset_id))
