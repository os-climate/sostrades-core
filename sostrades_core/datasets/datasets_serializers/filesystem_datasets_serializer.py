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
from typing import Any, Callable
import pandas as pd
import numpy as np
from os.path import join
import pickle as pkl


from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import JSONDatasetsSerializer
from sostrades_core.tools.tree.serializer import CSV_SEP
from sostrades_core.tools.tree.deserialization import isevaluatable


# Utility functions that mimic exactly the fashion in which api loads and saves dataframes from/to .csv
def _save_dataframe(file_path: str, df: pd.DataFrame) -> None:
    df.to_csv(file_path, sep=CSV_SEP, header=True, index=False)


def _load_dataframe(file_path: str) -> pd.DataFrame:
    df_value = pd.read_csv(file_path, na_filter=False)
    return df_value.applymap(isevaluatable)


def _save_pkl(file_path: str, obj: Any) -> None:
    with open(file_path, 'wb') as w:
        pkl.dump(obj, w)


def _load_pkl(file_path: str) -> Any:
    with open(file_path, 'rb') as r:
        return pkl.load(r)


class FileSystemDatasetsSerializer(JSONDatasetsSerializer):
    """
    Specific dataset serializer for dataset
    """
    # TYPE_IN_FILESYSTEM_PREFIXES = ['@dataframe@', '@array@']  # TODO: discuss
    TYPE_IN_FILESYSTEM_PARTICLE = '@'
    TYPE_DATAFRAME = 'dataframe'
    TYPE_ARRAY = 'array'
    TYPE_OBJECT = 'object'
    CSV_EXTENSION = 'csv'
    PKL_EXTENSION = 'pkl'
    EXTENSION_SEP = '.'
    TYPES_IN_FILESYSTEM = {TYPE_DATAFRAME, TYPE_ARRAY}

    def __init__(self):
        super().__init__()
        self.__logger = logging.getLogger(__name__)
        self.__current_dataset_directory = None

    def set_dataset_directory(self, dataset_directory):
        """
        Define the current dataset directory where specific data types will be serialized.
        """
        self.__current_dataset_directory = dataset_directory

    def convert_from_dataset_data(self, data_name:str, data_value:Any, data_types_dict:dict[str:str])-> Any:
        '''
        Convert data_value into data_type from the connector
        To be overridden for specific conversion.
        This function convert dataframe into dict and arrays into list, other types doesn't move.
        Can be used for json mapping for example.
        :param data_name: name of the data that is converted
        :type data_name: str
        :param data_value: value of the data that is converted
        :type data_value: Any
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        '''
        # sanity checks allowing not to load a @...@ into variable type not requiring filesystem storage.
        sanity = True
        filesystem_type = self.__get_filesystem_type(data_value)
        if data_name in data_types_dict:
            # unknown data type is handled in mother class method
            data_type = data_types_dict[data_name]

            # shortcut and load from pickle if an @object@ is encountered, in which case no need to check data type
            # this if clause is to be deleted when the subtype management based on descriptors is implemented(
            if filesystem_type == self.TYPE_OBJECT:
                self.__logger.debug(f"Loading {data_name} with filesystem descriptor {data_value} into the type "
                                    f"{data_type} required by the process.")
                return self._deserialize_object(data_value)

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
        Get the dataset descriptor type for a filesystem stored variable (e.g. dataframe, array)
        :param data_value: dataset descriptor value (e.g. @dataframe@d.csv)
        :type data_value: str
        :return: the dataset descriptor type (e.g. dataframe) or None if the variable does not have the @type@ prefix.
        """
        if isinstance(data_value, str):
            _tmp = data_value.split(self.TYPE_IN_FILESYSTEM_PARTICLE)
            if len(_tmp) >= 3:
                _fs_type = _tmp[1]
                if _fs_type in self.TYPES_IN_FILESYSTEM:
                    return _fs_type

    def __get_data_path(self, data_value: str) -> str:
        """
        Get the dataset descriptor path for a filesystem stored variable (e.g. dataframe, array) without sanity check.
        :param data_value: dataset descriptor value (e.g. @dataframe@d.csv)
        :type data_value: str
        :return: the dataset descriptor path (e.g. d.csv).
        """
        _tmp = data_value.split(self.TYPE_IN_FILESYSTEM_PARTICLE)
        _subpath = self.TYPE_IN_FILESYSTEM_PARTICLE.join(_tmp[2:])
        return _subpath

    def __deserialize_from_filesystem(self, deserialization_function: Callable[[str, ...], Any], descriptor_value: str,
                                      *args, **kwargs) -> Any:
        """
        Wrapper for a deserialization from filesystem function.
        :param deserialization_function: function to deserialize a given type taking the filesystem path.
        :type deserialization_function: callable
        :param descriptor_value: dataset descriptor value for the variable (e.g. "@dataframe@d.csv")
        :type descriptor_value: str
        :param args: for deserialization_function
        :param kwargs: for deserialization_function
        :return: the deserialized value for the variable
        """
        if self.__current_dataset_directory is None:
            self.__logger.warning(f"Error while trying to deserialize {descriptor_value} because dataset directory "
                                  f"is undefined")
            return descriptor_value
        else:
            data_subpath = self.__get_data_path(descriptor_value)
            data_path = join(self.__current_dataset_directory, data_subpath)
            return deserialization_function(data_path, *args, **kwargs)

    def __serialize_into_filesystem(self, serialization_function: Callable[[str, Any, ...], None], data_value: Any,
                                    descriptor_value: str, *args, **kwargs):
        """
        Wrapper for a serialization into filesystem function.
        :param serialization_function: function to serialize a given type taking the filesystem path and object.
        :type serialization_function: callable
        :param data_value: variable value
        :type data_value: Any
        :param descriptor_value: dataset descriptor value for the variable (e.g. "@dataframe@d.csv")
        :type descriptor_value: str
        :param args: for serialization_function
        :param kwargs: for serialization_function
        :return: dataset descriptor value for the variable (e.g. "@dataframe@d.csv")
        """
        if self.__current_dataset_directory is None:
            self.__logger.warning(f"Error while trying to serialize {data_value} because dataset directory "
                                  f"is undefined")
            return data_value
        else:
            # TODO: may need updating when datasets down to parameter level as assuming that the filename is linked to data name.
            _fname = self.__get_data_path(descriptor_value)
            data_path = join(self.__current_dataset_directory, _fname)
            serialization_function(data_path, data_value, *args, **kwargs)
            return descriptor_value

    def _deserialize_dataframe(self, data_value: str) -> pd.DataFrame:
        # NB: dataframe csv deserialization as in webapi
        return self.__deserialize_from_filesystem(_load_dataframe, data_value)

    def _deserialize_array(self, data_value: str) -> np.ndarray:
        # NB: to be improved with astype(subtype) along subtype management
        return self.__deserialize_from_filesystem(np.loadtxt, data_value)

    def _deserialize_object(self, data_value: str) -> Any:
        return self.__deserialize_from_filesystem(_load_pkl, data_value)

    def _serialize_dataframe(self, data_value: pd.DataFrame, data_name: str) -> str:
        # TODO: TEST whether api serialization methods suffice for dataframe nested types
        descriptor_value = self.EXTENSION_SEP.join((
            self.TYPE_IN_FILESYSTEM_PARTICLE.join(('', self.TYPE_DATAFRAME, data_name)),
            self.CSV_EXTENSION))
        # NB: dataframe csv serialization as in webapi
        return self.__serialize_into_filesystem(_save_dataframe, data_value, descriptor_value)

    def _serialize_array(self, data_value: np.ndarray, data_name: str) -> str:
        # TODO: [discuss] using pickle storage for anything that has dtype object
        if data_value.dtype == np.dtype('O'):
            return self._serialize_object(data_value, data_name)
        descriptor_value = self.EXTENSION_SEP.join((
            self.TYPE_IN_FILESYSTEM_PARTICLE.join(('', self.TYPE_ARRAY, data_name)),
            self.CSV_EXTENSION))
        # NB: converting ints to floats etc. to be improved along subtype management
        return self.__serialize_into_filesystem(np.savetxt, data_value, descriptor_value)

    def _serialize_object(self, data_value: Any, data_name: str) -> str:
        descriptor_value = self.EXTENSION_SEP.join((
            self.TYPE_IN_FILESYSTEM_PARTICLE.join(('', self.TYPE_OBJECT, data_name)),
            self.PKL_EXTENSION))
        return self.__serialize_into_filesystem(_save_pkl, data_value, descriptor_value)