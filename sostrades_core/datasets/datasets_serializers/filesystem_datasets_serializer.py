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
import pickle
import re
from os.path import join
from typing import Any, Callable

import numpy as np
import pandas as pd

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    DatasetDeserializeException,
    DatasetGenericException,
)
from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import (
    JSONDatasetsSerializer,
)
from sostrades_core.tools.folder_operations import makedirs_safe
from sostrades_core.tools.tree.deserialization import isevaluatable
from sostrades_core.tools.tree.serializer import CSV_SEP


# Utility functions that mimic exactly the fashion in which api loads and saves dataframes from/to .csv
def _save_dataframe(file_path: str, df: pd.DataFrame) -> None:
    """
    Save a DataFrame to a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        df (pd.DataFrame): The DataFrame to save.
    """
    df.to_csv(file_path, sep=CSV_SEP, header=True, index=False)


def _load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df_value = pd.read_csv(file_path, na_filter=False)
    return df_value.map(isevaluatable)


class FileSystemDatasetsSerializer(JSONDatasetsSerializer):
    """
    Specific dataset serializer for dataset
    """
    # TYPE_IN_FILESYSTEM_PREFIXES = ['@dataframe@', '@array@']  # TODO: discuss
    TYPE_IN_FILESYSTEM_PARTICLE = '@'
    TYPE_DATAFRAME = 'dataframe'
    TYPE_ARRAY = 'array'
    CSV_EXTENSION = 'csv'
    PKL_EXTENSION = 'pkl'
    EXTENSION_SEP = '.'
    TYPES_IN_FILESYSTEM = {TYPE_DATAFRAME, TYPE_ARRAY}

    # forbidden characters
    FORBIDDEN_CHARS_REGEX = r'[<>:\\/"\|\?\*]'
    FORBIDDEN_CHARS_END_OF_NAME = {" ", "."}
    FORBIDDEN_FS_NAMES = { "CON", "PRN", "AUX", "NUL",
                           "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                           "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
    PREFFIX_FORBIDDEN_FS_NAMES = "__"
    SUFFIX_END_OF_NAME = "_"

    # for the pickled data
    NON_SERIALIZABLE_PKL = 'non_serializable.pkl'
    TYPE_OBJECT = 'object'
    TYPE_OBJECT_IDENTIFIER = TYPE_IN_FILESYSTEM_PARTICLE.join(("", TYPE_OBJECT, ""))

    def __init__(self):
        super().__init__()
        self.__logger = logging.getLogger(__name__)
        self.__current_dataset_directory = None
        self.__pickle_data = {}

    @classmethod
    def format_filesystem_name(cls, fs_name: str) -> str:
        """
        Format a filesystem name so that it is compatible with Windows, Ubuntu and MacOS. Used here to format variable
        names for types that are serialized into the filesystem, as well as at connector level to format data group
        identifiers and to check the compliance of dataset identifiers.

        Args:
            fs_name (str): Filesystem name as defined by the user or namespace in case of wildcards.

        Returns:
            str: Formatted filesystem name filesystem-compatible using utf-8 encoding of forbidden characters.
        """
        # utf-8 replacement
        def replace_special_char(c):
            return f"_U{ord(c.group(0))}_"

        # replace forbidden characters by their replacement characters
        new_fs_name = re.sub(cls.FORBIDDEN_CHARS_REGEX, replace_special_char, fs_name)

        # replace end of name forbidden
        if new_fs_name and new_fs_name[-1] in cls.FORBIDDEN_CHARS_END_OF_NAME:
            new_fs_name = new_fs_name[:-1] + cls.SUFFIX_END_OF_NAME

        # replace forbidden names
        if new_fs_name in cls.FORBIDDEN_FS_NAMES:
            new_fs_name = cls.PREFFIX_FORBIDDEN_FS_NAMES + new_fs_name
        return new_fs_name

    def set_dataset_directory(self, dataset_directory: str) -> None:
        """
        Define the current dataset directory where specific data types will be serialized.

        Args:
            dataset_directory (str): The dataset directory path.
        """
        self.__current_dataset_directory = dataset_directory

    def read_descriptor_file(self, dataset_id: str, dataset_descriptor_path: str) -> dict[str, Any]:
        """
        Read the descriptor file for a dataset.

        Args:
            dataset_id (str): The dataset identifier.
            dataset_descriptor_path (str): The path to the dataset descriptor file.

        Returns:
            dict[str, Any]: The dataset descriptor.

        Raises:
            DatasetDeserializeException: If there is an error reading the descriptor file.
        """
        # Load the descriptor, the serializer loads the pickle if it exists
        self.check_path_exists("Dataset descriptor file", dataset_descriptor_path)
        dataset_descriptor = None
        try:
            with open(dataset_descriptor_path, "r", encoding="utf-8") as file:
                dataset_descriptor = json.load(fp=file)
        except TypeError as exception:
            raise DatasetDeserializeException(dataset_id, f'type error exception in dataset descriptor, {str(exception)}')
        except json.JSONDecodeError as exception:
            raise DatasetDeserializeException(dataset_id, f'dataset descriptor does not have a valid json format, {str(exception.msg)}')

        return dataset_descriptor

    def write_descriptor_file(self, dataset_descriptor_path: str, dataset_descriptor: dict[str, Any]) -> None:
        """
        Write the dataset descriptor to a file.

        Args:
            dataset_descriptor_path (str): The path to the dataset descriptor file.
            dataset_descriptor (dict[str, Any]): The dataset descriptor.
        """
        # read the already existing values
        # write in dataset descriptor
        self.check_path_exists("Dataset descriptor file", dataset_descriptor_path)
        with open(dataset_descriptor_path, "w", encoding="utf-8") as file:
                json.dump(obj=dataset_descriptor, fp=file, indent=4)

    def check_path_exists(self, directory_info: str, dataset_path: str) -> bool:
        """
        Check that the dataset directory exists if not raise an error.

        Args:
            directory_info (str): Info of what directory it is (dataset db, dataset...).
            dataset_path (str): The path to the dataset directory.

        Returns:
            bool: True if the path exists, False otherwise.

        Raises:
            DatasetGenericException: If the path does not exist.
        """
        if not os.path.exists(dataset_path):
            raise DatasetGenericException(f"{directory_info} not found at {dataset_path}.")

    def check_folder_name(self, folder_name: str) -> bool:
        """
        Check if the folder name is valid.

        Args:
            folder_name (str): The folder name to check.

        Returns:
            bool: True if the folder name is valid, False otherwise.
        """
        filesystem_dataset_identifier = self.format_filesystem_name(folder_name)
        if filesystem_dataset_identifier != folder_name:
            return False
        return True

    def load_pickle_data(self) -> None:
        """
        Load and buffer the non-serializable types data stored in the pickle file.
        """
        pkl_data = {}
        if self.__current_dataset_directory is None:
            self.__logger.error("Error while trying to load pickled data because dataset directory is undefined")
        else:
            if not os.path.exists(self.__current_dataset_directory):
                makedirs_safe(self.__current_dataset_directory, exist_ok=True)
            non_serializable_pkl_path = os.path.join(self.__current_dataset_directory, self.NON_SERIALIZABLE_PKL)
            if os.path.exists(non_serializable_pkl_path):
                try:
                    with open(non_serializable_pkl_path, 'rb') as pkl_file:
                        pkl_data = pickle.load(pkl_file)
                except TypeError as exception:
                    self.__logger.error(f'Type error exception in dataset pickle file for non-serializable data, {str(exception)}')
                except pickle.UnpicklingError as exception:
                    self.__logger.error(f'Dataset pickle file for non-serializable data does not have a valid pickle format, {str(exception)}')
        self.__pickle_data = pkl_data

    def clear_pickle_data(self) -> None:
        """
        Clear buffered data from pickle load.
        """
        self.__pickle_data = {}

    def dump_pickle_data(self) -> None:
        """
        Write the buffered non-serializable data into the dataset pickle file and clear the buffer.
        """
        if self.__pickle_data:
            if self.__current_dataset_directory is None:
                self.__logger.error("Error while trying to dump pickled data because dataset directory is undefined")
            else:
                non_serializable_pkl_path = os.path.join(self.__current_dataset_directory, self.NON_SERIALIZABLE_PKL)
                with open(non_serializable_pkl_path, 'wb') as pkl_file:
                    pickle.dump(self.__pickle_data, pkl_file)
        self.clear_pickle_data()

    def __clean_from_pickle_data(self, data_name: str) -> None:
        """
        Utility method used to clear an entry from the buffered pickle data, in order to make sure that regularly
        serialized data are not repeated in the pickle too.

        Args:
            data_name (str): The name of the data to clear.
        """
        # TODO: [discuss] is this necessary ?
        if data_name in self.__pickle_data:
            del self.__pickle_data[data_name]

    def convert_from_dataset_data(self, data_name: str, data_value: Any, data_types_dict: dict[str, str]) -> Any:
        """
        Convert data_value into data_type from the connector.

        Args:
            data_name (str): Name of the data that is converted.
            data_value (Any): Value of the data that is converted.
            data_types_dict (dict[str, str]): Dict of data types {name: type}.

        Returns:
            Any: The converted data value.
        """
        # sanity checks allowing not to load a @...@ into variable type not requiring filesystem storage.
        sanity = True
        filesystem_type = self.__get_filesystem_type(data_value)
        if data_name in data_types_dict:
            # unknown data type is handled in mother class method
            data_type = data_types_dict[data_name]
            # TODO[discuss]: when the data is in the pickle pre-fill with @object@ and will be overwritten by connector
            if filesystem_type == self.TYPE_OBJECT:
                self.__logger.debug(f"{data_name} with filesystem descriptor {data_value} is loaded from"
                                    f"non-serializable types pickle file.")
                return self.__deserialize_object(data_value)

            # insane if there is an @...@ descriptor that is unknown or mismatching
            elif filesystem_type is not None and filesystem_type != data_type:
                sanity = False
                self.__logger.warning(f"Error while trying to load {data_name} with filesystem descriptor "
                                      f"{data_value} into the type {data_type} required by the process. Types"
                                      f"supported for storage in filesystem are {self.TYPES_IN_FILESYSTEM}")

            # insane if there is no @...@ for a data type expecting it
            elif filesystem_type is None and data_type in self.TYPES_IN_FILESYSTEM:
                sanity = False
                self.__logger.warning(f"Error while trying to load {data_name} with value "
                                      f"{data_value} into the type {data_type}. Types"
                                      f"requiring storage in filesystem are {self.TYPES_IN_FILESYSTEM}")
        if sanity:
            return super().convert_from_dataset_data(data_name=data_name,
                                                     data_value=data_value,
                                                     data_types_dict=data_types_dict)
        else:
            return data_value

    def __get_filesystem_type(self, data_value: str) -> str:
        """
        Get the dataset descriptor type for a filesystem stored variable (e.g. dataframe, array).

        Args:
            data_value (str): Dataset descriptor value (e.g. @dataframe@d.csv).

        Returns:
            str: The dataset descriptor type (e.g. dataframe) or None if the variable does not have the @type@ prefix.
        """
        if isinstance(data_value, str):
            _tmp = data_value.split(self.TYPE_IN_FILESYSTEM_PARTICLE)
            if len(_tmp) >= 3:
                _fs_type = _tmp[1]
                if _fs_type in self.TYPES_IN_FILESYSTEM or _fs_type == self.TYPE_OBJECT:
                    return _fs_type

    def __get_data_path(self, data_value: str) -> str:
        """
        Get the dataset descriptor path for a filesystem stored variable (e.g. dataframe, array) without sanity check.

        Args:
            data_value (str): Dataset descriptor value (e.g. @dataframe@d.csv).

        Returns:
            str: The dataset descriptor path (e.g. d.csv).
        """
        _tmp = data_value.split(self.TYPE_IN_FILESYSTEM_PARTICLE)
        _subpath = self.TYPE_IN_FILESYSTEM_PARTICLE.join(_tmp[2:])
        return _subpath

    def __deserialize_from_filesystem(self, deserialization_function: Callable[[str, ...], Any], descriptor_value: str,
                                      *args, **kwargs) -> Any:
        """
        Wrapper for a deserialization from filesystem function.

        Args:
            deserialization_function (Callable[[str, ...], Any]): Function to deserialize a given type taking the filesystem path.
            descriptor_value (str): Dataset descriptor value for the variable (e.g. "@dataframe@d.csv").
            *args: Additional arguments for deserialization_function.
            **kwargs: Additional keyword arguments for deserialization_function.

        Returns:
            Any: The deserialized value for the variable.
        """
        if self.__current_dataset_directory is None:
            self.__logger.error(f"Error while trying to deserialize {descriptor_value} because dataset directory "
                                f"is undefined")
            return descriptor_value
        else:
            data_subpath = self.__get_data_path(descriptor_value)
            data_path = join(self.__current_dataset_directory, data_subpath)
            return deserialization_function(data_path, *args, **kwargs)

    def __serialize_into_filesystem(self, serialization_function: Callable[[str, Any, ...], None], data_value: Any,
                                    data_name: str, descriptor_value: str, *args, **kwargs) -> str:
        """
        Wrapper for a serialization into filesystem function.

        Args:
            serialization_function (Callable[[str, Any, ...], None]): Function to serialize a given type taking the filesystem path and object.
            data_value (Any): Variable value.
            data_name (str): Name of the data.
            descriptor_value (str): Dataset descriptor value for the variable (e.g. "@dataframe@d.csv").
            *args: Additional arguments for serialization_function.
            **kwargs: Additional keyword arguments for serialization_function.

        Returns:
            str: Dataset descriptor value for the variable (e.g. "@dataframe@d.csv").
        """
        if self.__current_dataset_directory is None:
            self.__logger.error(f"Error while trying to serialize {data_value} because dataset directory "
                                f"is undefined")
            return data_value
        else:
            if not os.path.exists(self.__current_dataset_directory):
                makedirs_safe(self.__current_dataset_directory, exist_ok=True)
            _fname = self.__get_data_path(descriptor_value)
            data_path = join(self.__current_dataset_directory, _fname)
            try:
                serialization_function(data_path, data_value, *args, **kwargs)
                return descriptor_value
            except Exception as exception:  # at least TypeError, ValueError for arrays
                self.__logger.debug(f"{descriptor_value} will be stored in pickle after serialization exception:"
                                    f"\n {exception}")
                if os.path.exists(data_path):
                    os.remove(data_path)
                return self.__serialize_object(data_value, data_name)

    def _deserialize_dataframe(self, data_value: str, data_name: str = None) -> pd.DataFrame:
        """
        Deserialize a DataFrame from the filesystem.

        Args:
            data_value (str): Dataset descriptor value for the DataFrame.
            data_name (str, optional): Name of the data. Defaults to None.

        Returns:
            pd.DataFrame: The deserialized DataFrame.
        """
        # NB: dataframe csv deserialization as in webapi
        return self.__deserialize_from_filesystem(_load_dataframe, data_value)

    def _deserialize_array(self, data_value: str) -> np.ndarray:
        """
        Deserialize an array from the filesystem.

        Args:
            data_value (str): Dataset descriptor value for the array.

        Returns:
            np.ndarray: The deserialized array.
        """
        # NB: to be improved with astype(subtype) along subtype management
        return self.__deserialize_from_filesystem(np.loadtxt, data_value, ndmin=1)

    def _serialize_dataframe(self, data_value: pd.DataFrame, data_name: str) -> str:
        """
        Serialize a DataFrame into the filesystem.

        Args:
            data_value (pd.DataFrame): The DataFrame to serialize.
            data_name (str): Name of the data.

        Returns:
            str: Dataset descriptor value for the DataFrame.
        """
        descriptor_value = self.EXTENSION_SEP.join((
            self.TYPE_IN_FILESYSTEM_PARTICLE.join(('', self.TYPE_DATAFRAME, self.format_filesystem_name(data_name))),
            self.CSV_EXTENSION))
        # NB: dataframe csv serialization as in webapi
        return self.__serialize_into_filesystem(_save_dataframe, data_value, data_name, descriptor_value)

    def _serialize_array(self, data_value: np.ndarray, data_name: str) -> str:
        """
        Serialize an array into the filesystem.

        Args:
            data_value (np.ndarray): The array to serialize.
            data_name (str): Name of the data.

        Returns:
            str: Dataset descriptor value for the array.
        """
        descriptor_value = self.EXTENSION_SEP.join((
            self.TYPE_IN_FILESYSTEM_PARTICLE.join(('', self.TYPE_ARRAY, self.format_filesystem_name(data_name))),
            self.CSV_EXTENSION))
        # NB: converting ints to floats etc. to be improved along subtype management
        return self.__serialize_into_filesystem(np.savetxt, data_value, data_name, descriptor_value)

    def _serialize_jsonifiable(self, data_value: Any, data_name: str) -> Any:
        """
        Serialize a JSONifiable data value.

        Args:
            data_value (Any): The data value to serialize.
            data_name (str): Name of the data.

        Returns:
            Any: The serialized data value.
        """
        try:
            _ = json.dumps(data_value)
            self.__clean_from_pickle_data(data_name)
            return data_value
        except TypeError:  # non-jsonifiable
            self.__logger.debug(f"{data_name} to be stored in pickle because non-jsonifiable")
            return self.__serialize_object(data_value, data_name)

    def __serialize_object(self, data_value: Any, data_name: str) -> str:
        """
        Serialize an object into the pickle file.

        Args:
            data_value (Any): The object to serialize.
            data_name (str): Name of the data.

        Returns:
            str: Dataset descriptor value for the object.
        """
        descriptor_value = self.TYPE_IN_FILESYSTEM_PARTICLE.join(('', self.TYPE_OBJECT, data_name))
        self.__pickle_data[data_name] = data_value
        return descriptor_value

    def __deserialize_object(self, data_value: str) -> Any:
        """
        Deserialize an object from the pickle file.

        Args:
            data_value (str): Dataset descriptor value for the object.

        Returns:
            Any: The deserialized object.
        """
        pickle_key = self.__get_data_path(data_value)
        return self.__pickle_data[pickle_key]
