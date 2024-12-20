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

import abc
import logging
from typing import Any


class AbstractDatasetsSerializer(abc.ABC):
    """
    Abstract class to inherit in order to build specific datasets connector
    """
    __logger = logging.getLogger(__name__)
    SOSTRADES_TYPES = {'string', 'int', 'float', 'bool', 'list', 'dict', 'dataframe', 'array'}

    @abc.abstractmethod
    def convert_from_dataset_data(self, data_name: str, data_value: Any, data_types_dict: dict[str, str]) -> Any:
        """
        Convert data_value into data_type from the dataset connector.
        To be overridden for specific conversion.
        This function converts dataframe into dict and arrays into list, other types don't move.
        Can be used for json mapping for example.

        Args:
            data_name (str): Name of the data that is converted.
            data_value (Any): Value of the data that is converted.
            data_types_dict (dict[str, str]): Dict of data types {name: type}.

        Returns:
            Any: Converted data value.
        """

    @abc.abstractmethod
    def convert_to_dataset_data(self, data_name: str, data_value: Any, data_types_dict: dict[str, str]) -> Any:
        """
        Convert data_value into dataset connector format.

        Args:
            data_name (str): Name of the data that is converted.
            data_value (Any): Value of the data that is converted.
            data_types_dict (dict[str, str]): Dict of data types {name: type}.

        Returns:
            Any: Converted data value.
        """
