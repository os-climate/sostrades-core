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

from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import JSONDatasetsSerializer


class FileSystemDatasetsSerializer(JSONDatasetsSerializer): # FIXME: REWRITE without inheritance
    """
    Specific dataset serializer for dataset
    """
    # TYPE_IN_FILESYSTEM_PREFIXES = ['@dataframe@', '@array@']  # TODO: discuss
    TYPE_IN_FILESYSTEM_PARTICLE = '@'
    TYPES_IN_FILESYSTEM = {'dataframe', 'array'}

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
        filesystem_type = self._get_filesystem_type(data_value)
        if data_name in data_types_dict:
            # unknown data type is handled in mother class method
            data_type = data_types_dict[data_name]
            # insane if there is an @...@ descriptor that is unknown or mismatching
            if filesystem_type is not None and (filesystem_type != data_type or
                                                filesystem_type not in self.TYPES_IN_FILESYSTEM):
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

    def _get_filesystem_type(self, data_value):
        filesystem_type = None
        if isinstance(data_value, str):
            _tmp = data_value.split(self.TYPE_IN_FILESYSTEM_PARTICLE)
            if len(_tmp) > 3:
                filesystem_type = _tmp[1]
        return filesystem_type

    def _deserialize_dataframe(self, data_value):
        return NotImplementedError()

    def _deserialize_array(self, data_value):
        return NotImplementedError()

    def _serialize_dataframe(self, data_value, dump_file_name):
        # TODO: the conector needs to oversee the dump in order to provide meaningful file names
        return NotImplementedError()

    def _serialize_array(self, data_value, dump_file_name):
        # TODO: the conector needs to oversee the dump in order to provide meaningful file names
        return NotImplementedError()
