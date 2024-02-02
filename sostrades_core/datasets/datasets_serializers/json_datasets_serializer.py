"""
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
"""

from typing import Any, List
import pandas as pd
import numpy as np

from sostrades_core.datasets.datasets_serializers.abstract_datasets_serializer import AbstractDatasetsSerializer


class JSONDatasetsSerializer(AbstractDatasetsSerializer):
    """
    Specific dataset serializer for dataset in json format
    """

    def convert_from_dataset_data(self, data_name:str, data_value:Any, data_types_dict:dict[str:str])-> Any:
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

        converted_data = ""
        try:
            if data_type in ['string', 'int', 'float', 'bool', 'list', 'dict']:
                converted_data = data_value
            elif data_type == 'dataframe':
                converted_data = pd.DataFrame.from_dict(data_value)
            elif data_type == 'array':
                converted_data = np.array(data_value)
            else:
                converted_data = data_value
                self.__logger.warning(f"Data type {data_type} for data {data_name} not found in default type list 'string', 'int', 'float', 'bool', 'list', 'dict', 'dataframe, 'array'.")
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")

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
            if data_type in ['string', 'int', 'float', 'bool', 'list', 'dict']:
                converted_data = data_value
            elif data_type == 'dataframe':
                # convert dataframe into dict with orient='list' to have {column:values}
                converted_data = pd.DataFrame.to_dict(data_value,'list')
            elif data_type == 'array':
                converted_data = list(data_value)
            else:
                converted_data = data_value
                self.__logger.warning(f"Data type {data_type} for data {data_name} not found in default type list 'string', 'int', 'float', 'bool', 'list', 'dict', 'dataframe, 'array'.")
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")

        return converted_data
    