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
from typing import Any
import pandas as pd
import numpy as np
from os.path import join

from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import JSONDatasetsSerializer


class FileSystemDatasetsSerializer(JSONDatasetsSerializer):
    """
    Specific dataset serializer for dataset
    """
    # TYPE_IN_FILESYSTEM_PREFIXES = ['@dataframe@', '@array@']  # TODO: discuss
    TYPE_IN_FILESYSTEM_PARTICLE = '@'
    TYPE_DATAFRAME = 'dataframe'
    TYPE_ARRAY = 'array'
    EXTENSION = 'csv'
    EXTENSION_SEP = '.'
    TYPES_IN_FILESYSTEM = {TYPE_DATAFRAME, TYPE_ARRAY}

    def __init__(self):
        super().__init__()
        self.__current_dataset_directory = None

    def set_dataset_directory(self, dataset_directory):
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
            # insane if there is an @...@ descriptor that is unknown or mismatching
            if filesystem_type is not None and filesystem_type != data_type:
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

    def __get_filesystem_type(self, data_value):
        if isinstance(data_value, str):
            _tmp = data_value.split(self.TYPE_IN_FILESYSTEM_PARTICLE)
            if len(_tmp) > 3:
                _fs_type = _tmp[1]
                if _fs_type in self.TYPES_IN_FILESYSTEM:
                    return _fs_type

    def __get_data_path(self, data_value):
        _tmp = data_value.split(self.TYPE_IN_FILESYSTEM_PARTICLE)
        _subpath = self.TYPE_IN_FILESYSTEM_PARTICLE.join(_tmp[2:])
        return _subpath

    def __deserialize_from_filesystem(self, deserialization_function, data_value, *args, **kwargs):
        if self.__current_dataset_directory is None:
            self.__logger.warning(f"Error while trying to deserialize {data_value} because dataset directory "
                                  f"is undefined")
            return data_value
        else:
            data_subpath = self.__get_data_path(data_value)
            data_path = join(self.__current_dataset_directory, data_subpath)
            return deserialization_function(data_path, *args, **kwargs)

    def _deserialize_dataframe(self, data_value):
        return self.__deserialize_from_filesystem(pd.read_csv, data_value)

    def _deserialize_array(self, data_value):
        return self.__deserialize_from_filesystem(np.loadtxt, data_value)

    def __serialize_into_filesystem(self, serialization_function, data_value, data_name, *args, **kwargs):
        if self.__current_dataset_directory is None:
            self.__logger.warning(f"Error while trying to serialize {data_value} because dataset directory "
                                  f"is undefined")
            return data_value
        else:
            # TODO: may need updating when datasets down to parameter level
            data_subpath = self.EXTENSION_SEP.join((data_name, self.EXTENSION))
            data_path = join(self.__current_dataset_directory, data_subpath)
            serialization_function(data_path, data_value, *args, **kwargs)

    def __dump_dataframe(self, dump_path, df, *args, **kwargs):
        df.to_csv(dump_path, *args, **kwargs)

    def _serialize_dataframe(self, data_value, data_name):
        self.__serialize_into_filesystem(self.__dump_dataframe, data_value, data_name)
        # TODO: may need updating when datasets down to parameter level
        return self.TYPE_IN_FILESYSTEM_PARTICLE.join(('', self.TYPE_DATAFRAME, data_name))

    def _serialize_array(self, data_value, data_name):
        self.__serialize_into_filesystem(np.savetxt, data_value, data_name)
        # TODO: may need updating when datasets down to parameter level
        return self.TYPE_IN_FILESYSTEM_PARTICLE.join(('', self.TYPE_ARRAY, data_name))
