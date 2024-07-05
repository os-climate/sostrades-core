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
from typing import Any

import numpy as np

from sostrades_core.datasets.datasets_serializers.json_datasets_serializer import (
    JSONDatasetsSerializer,
)


class BigQueryDatasetsSerializer(JSONDatasetsSerializer):
    """
    Specific dataset serializer for dataset in json format
    """
    VALUE = "value"

    def __init__(self):
        super().__init__()
        self.__logger = logging.getLogger(__name__)

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
                converted_data = self._deserialize_list(data_value)
            else:
                converted_data = super().convert_from_dataset_data(data_name, data_value, data_types_dict)
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(
                f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")
        return converted_data

    def convert_to_dataset_data(self, data_name:str, data_value:Any, data_types_dict:dict[str:str])-> Any:
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
                converted_data = self._serialize_list(data_value, data_name)
            else:
                converted_data = super().convert_to_dataset_data(data_name, data_value, data_types_dict)
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(
                f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")
        return converted_data

    def _serialize_dataframe(self, data_value, data_name):
        return data_value

    def _deserialize_dataframe(self, data_value):
        return data_value

    def _serialize_list(self, data_value, data_name):
        return {self.VALUE: self._serialize_sub_element_jsonifiable(data_value)}

    def _deserialize_list(self, data_value: dict[str:list]) -> np.ndarray:
        return list(data_value[self.VALUE])

    def _serialize_array(self, data_value, data_name):
        return {self.VALUE: self._serialize_sub_element_jsonifiable(data_value.tolist())}

    def _deserialize_array(self, data_value: dict[str:list]) -> np.ndarray:
        return np.array(data_value[self.VALUE])

    def _serialize_sub_element_jsonifiable(self, data_value:list):
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
