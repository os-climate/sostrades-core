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
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from sostrades_core.datasets.datasets_serializers.abstract_datasets_serializer import (
    AbstractDatasetsSerializer,
)


class JSONDatasetsSerializer(AbstractDatasetsSerializer):
    """Specific dataset serializer for dataset in json format"""

    def __init__(self):
        super().__init__()
        self.__logger = logging.getLogger(__name__)

    def convert_from_dataset_data(self, data_name: str, data_value: Any, data_types_dict: dict[str, str]) -> Any:
        """
        Convert data_value into data_type from the connector.
        This function converts dataframe into dict and arrays into list, other types don't move.
        Can be used for json mapping for example.

        Args:
            data_name (str): name of the data that is converted
            data_value (Any): value of the data that is converted
            data_types_dict (dict[str, str]): dict of data types {name: type}

        Returns:
            Any: Converted data

        """
        # retrieve the type of the data into the data_type_dict.
        # If the data type is not found, the data value is not converted
        data_type = None
        if data_name in data_types_dict.keys():
            data_type = data_types_dict[data_name]

        converted_data = ""
        try:
            if data_type in self.SOSTRADES_TYPES:
                if data_value is None:
                    converted_data = data_value
                elif data_type == 'dataframe':
                    converted_data = self._deserialize_dataframe(data_value, data_name)
                elif data_type == 'array':
                    converted_data = self._deserialize_array(data_value)
                else:
                    converted_data = data_value
            else:
                self.__logger.warning(f"Data type {data_type} for data {data_name} not found in default type list 'string', 'int', 'float', 'bool', 'list', 'dict', 'dataframe, 'array'.")
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")

        return converted_data

    def convert_to_dataset_data(self, data_name: str, data_value: Any, data_types_dict: dict[str, str]) -> Any:
        """
        Convert data_value into connector format.

        Args:
            data_name (str): name of the data that is converted
            data_value (Any): value of the data that is converted
            data_types_dict (dict[str, str]): dict of data types {name: type}

        Returns:
            Any: Converted data

        """
        # retrieve the type of the data into the data_type_dict.
        # If the data type is not found, the data value is not converted
        data_type = None
        if data_name in data_types_dict.keys():
            data_type = data_types_dict[data_name]

        converted_data = ""
        try:
            if data_type in self.SOSTRADES_TYPES:
                if data_value is None:
                    converted_data = data_value
                elif data_type == 'dataframe':
                    # convert dataframe into dict with orient='list' to have {column:values}
                    converted_data = self._serialize_dataframe(data_value, data_name)
                elif data_type == 'array':
                    converted_data = self._serialize_array(data_value, data_name)
                else:
                    converted_data = self._serialize_jsonifiable(data_value, data_name)
            else:
                self.__logger.warning(f"Data type {data_type} for data {data_name} not found in default type list 'string', 'int', 'float', 'bool', 'list', 'dict', 'dataframe, 'array'.")
        except Exception as error:
            converted_data = data_value
            self.__logger.warning(f"Error while trying to convert data {data_name} with value {data_value} into the type {data_type}: {error}")
        return converted_data

    def _deserialize_dataframe(self, data_value: dict, data_name: str) -> pd.DataFrame:
        """
        Deserialize dataframe from dict.

        Args:
            data_value (dict): Data to be deserialized
            data_name (str): Name of the data

        Returns:
            pd.DataFrame: Deserialized dataframe

        """
        return pd.DataFrame.from_dict(data_value)

    def _deserialize_array(self, data_value: list) -> np.ndarray:
        """
        Deserialize array from list.

        Args:
            data_value (list): Data to be deserialized

        Returns:
            np.ndarray: Deserialized array

        """
        return np.array(data_value)

    def _serialize_dataframe(self, data_value: pd.DataFrame, data_name: str) -> dict:
        """
        Serialize dataframe to dict.

        Args:
            data_value (pd.DataFrame): Data to be serialized
            data_name (str): Name of the data

        Returns:
            dict: Serialized dataframe

        """
        return pd.DataFrame.to_dict(data_value, 'list')

    def _serialize_array(self, data_value: np.ndarray, data_name: str) -> list:
        """
        Serialize array to list.

        Args:
            data_value (np.ndarray): Data to be serialized
            data_name (str): Name of the data

        Returns:
            list: Serialized array

        """
        return data_value.tolist()

    def _serialize_jsonifiable(self, data_value: Any, data_name: str) -> Any:
        """
        Serialize jsonifiable data.

        Args:
            data_value (Any): Data to be serialized
            data_name (str): Name of the data

        Returns:
            Any: Serialized data

        """
        return data_value
