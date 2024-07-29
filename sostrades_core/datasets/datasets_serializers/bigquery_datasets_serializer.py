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
from string import ascii_letters, digits
from typing import Any

import numpy as np

from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import (
    JSONDatasetsSerializer,
)


class BigQueryDatasetsSerializer(JSONDatasetsSerializer):
    """
    Specific dataset serializer for dataset in json format
    """
    LIST_VALUE = "value"  # this is for storage of list/array as dict, for visibility on bigquery
    COL_NAME_ALLOWED_CHAR_0 = set(ascii_letters + '_')
    COL_NAME_ALLOWED_CHAR_1 = set(digits)
    REPLACEMENT_FORBIDDEN_CHAR = "_"

    def __init__(self):
        super().__init__()
        self.__logger = logging.getLogger(__name__)
        self.__col_name_index = None

    def set_col_name_index(self, col_name_index):
        self.__col_name_index = col_name_index

    def clear_col_name_index(self):
        self.__col_name_index = None

    @property
    def col_name_index(self):
        return self.__col_name_index

    def __format_bigquery_col_name(self, col_name: str) -> (str, str):
        """
        Returns a column name that is compatible with bigquery as well as the old colum name to index or None if no change
        :param col_name: original column name in the variable
        :return: bigquery compatible col_name, old name to index or None
        """
        bq_col_name = ""
        for i, _char in enumerate(col_name):
            if _char in self.COL_NAME_ALLOWED_CHAR_0 or i > 0 and _char in self.COL_NAME_ALLOWED_CHAR_1:
                bq_col_name += _char
            else:
                bq_col_name += self.REPLACEMENT_FORBIDDEN_CHAR
        return bq_col_name

    def convert_from_dataset_data(self, data_name: str, data_value: Any, data_types_dict: dict[str:str]) -> Any:
        '''
        Convert data_value into data_type from the connector
        To be override for specific conversion.
        This function convert dataframe into dict and arrays into list, other types doesn't move.
        Can be used for json mapping for example.
        :param data_name: name of the data that is converted
        :type data_name: str
        :param data_value: value of the data that is converted
        :type data_value: Any
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        '''
        # retreive the type of the data into the data_type_dict.
        # If the data type os not found, the data value is not converted
        data_type = None
        if data_name in data_types_dict.keys():
            data_type = data_types_dict[data_name]

        try:
            if data_type == "list":
                converted_data = self.__deserialize_list(data_value)
            elif data_type == 'dict':
                converted_data = self.__deserialize_dict(data_value, data_name)
            else:
                converted_data = super().convert_from_dataset_data(data_name, data_value, data_types_dict)
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(
                f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")
        return converted_data

    def convert_to_dataset_data(self, data_name: str, data_value: Any, data_types_dict: dict[str:str]) -> Any:
        '''
        Convert data_value into connector format
        :param data_name: name of the data that is converted
        :type data_name: str
        :param data_value: value of the data that is converted
        :type data_value: Any
        :param data_types_dict: dict of data types {name: type}
        :type data_types_dict: dict[str:str]
        '''
        # retreive the type of the data into the data_type_dict.
        # If the data type os not found, the data value is not converted
        data_type = None
        if data_name in data_types_dict.keys():
            data_type = data_types_dict[data_name]

        converted_data = ""
        try:
            if data_type == 'list':
                converted_data = self.__serialize_list(data_value)
            elif data_type == 'dict':
                converted_data = self.__serialize_dict(data_value, data_name)
            else:
                converted_data = super().convert_to_dataset_data(data_name, data_value, data_types_dict)
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(
                f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")
        return converted_data

    def _serialize_dataframe(self, data_value, data_name):
        _renamer = dict()
        _all_new_cols = set()
        for old_col_name in data_value.columns:
            new_col_name = self.__format_bigquery_col_name(old_col_name)
            new_index = self.__col_name_index.get(data_name, dict())
            new_index.update({new_col_name: old_col_name})
            self.__col_name_index[data_name] = new_index
            _renamer[old_col_name] = new_col_name
            if new_col_name in _all_new_cols:
                self.__logger.error(f"Duplicate BigQuery-compatible column name {new_col_name} for dataframe "
                                    f"{data_name}, probable data corruption due to column overwrite.")
            _all_new_cols.add(new_col_name)
        return data_value.rename(columns=_renamer, copy=True)

    def _deserialize_dataframe(self, data_value, data_name):
        return data_value.rename(columns=self.__col_name_index[data_name])

    def _serialize_array(self, data_value, data_name):
        return {self.LIST_VALUE: self._serialize_sub_element_jsonifiable(data_value.tolist())}

    def _deserialize_array(self, data_value: dict[str:list]) -> np.ndarray:
        return np.array(data_value[self.LIST_VALUE])

    def __serialize_list(self, data_value):
        return {self.LIST_VALUE: self._serialize_sub_element_jsonifiable(data_value)}

    def __deserialize_list(self, data_value: dict[str:list]) -> np.ndarray:
        return list(data_value[self.LIST_VALUE])

    def __serialize_dict(self, data_value, data_name):
        _renamer = dict()
        _all_new_cols = set()
        for old_key in data_value.keys():
            new_key = self.__format_bigquery_col_name(old_key)
            new_index = self.__col_name_index.get(data_name, dict())
            new_index.update({new_key: old_key})
            self.__col_name_index[data_name] = new_index
            _renamer[old_key] = new_key
            if new_key in _all_new_cols:
                self.__logger.error(f"Duplicate BigQuery-compatible column name {new_key} for dictionary "
                                    f"{data_name}, probable data corruption due to key overwrite.")
            _all_new_cols.add(new_key)
        return self.__rename_dict_keys(data_value, _renamer)

    def __deserialize_dict(self, data_value, data_name):
        return self.__rename_dict_keys(data_value, self.__col_name_index[data_name])

    def __rename_dict_keys(self, dict_to_rename_keys, key_name_map):
        return {(key_name_map[key] if key in key_name_map else key): value
                for key, value in dict_to_rename_keys.items()}

    def _serialize_sub_element_jsonifiable(self, data_value: list):
        '''
        convert sub element of a list into non numpy format
        '''
        json_value = []
        for element in data_value:
            if isinstance(element, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):

               json_value.append(int(element))

            elif isinstance(element, (np.float_, np.float16, np.float32, np.float64)):
                json_value.append(float(element))
            else:
                json_value.append(element)
        return json_value
