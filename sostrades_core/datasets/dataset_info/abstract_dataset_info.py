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
from dataclasses import dataclass
from typing import ClassVar, Dict, List


class DatasetsInfoMappingException(Exception):
    """Generic exception for dataset info"""

    pass


@dataclass(frozen=True)
class AbstractDatasetInfo(abc.ABC):
    """Stores the information of a dataset"""

    # Keys for parsing json
    VERSION_ID_KEY: ClassVar[str] = "version_id"
    CONNECTOR_ID_KEY: ClassVar[str] = "connector_id"
    DATASET_ID_KEY: ClassVar[str] = "dataset_id"
    PARAMETER_ID_KEY: ClassVar[str] = "parameter_name"

    WILDCARD: ClassVar[str] = "*"
    SEPARATOR: ClassVar[str] = '|'

    # Id of the connector
    connector_id: str
    # Dataset id for this connector
    dataset_id: str

    @property
    @abc.abstractmethod
    def version_id(self) -> str:
        """Abstract property to be overridden in each subclass to return the version id."""

    @property
    def dataset_info_id(self) -> str:
        """
        Returns the dataset info id by joining version_id, connector_id, and dataset_id with a separator.

        Returns:
            str: The dataset info id.

        """
        return self.get_mapping_id([self.version_id, self.connector_id, self.dataset_id])

    @staticmethod
    def get_mapping_id(ids: List[str]) -> str:
        """
        Joins the given list of ids with a separator to form a mapping id.

        Args:
            ids (List[str]): List of ids to join.

        Returns:
            str: The joined mapping id.

        """
        return AbstractDatasetInfo.SEPARATOR.join(ids)

    @staticmethod
    @abc.abstractmethod
    def deserialize(dataset_mapping_key: str) -> Dict[str, str]:
        """
        Abstract method to deserialize a dataset mapping key.

        Args:
            dataset_mapping_key (str): The dataset mapping key to deserialize.

        Returns:
            Dict[str, str]: The deserialized dataset information.

        """

    @staticmethod
    @abc.abstractmethod
    def create(input_dict: Dict[str, str]) -> AbstractDatasetInfo:
        """
        Abstract method to create an instance of AbstractDatasetInfo.

        Args:
            input_dict (Dict[str, str]): The input dictionary containing dataset information.

        Returns:
            AbstractDatasetInfo: The created instance of AbstractDatasetInfo.

        """

    @abc.abstractmethod
    def copy_with_new_ns(self, associated_namespace: str) -> AbstractDatasetInfo:
        """
        Abstract method to create a new instance of AbstractDatasetInfo with a new namespace.

        Args:
            associated_namespace (str): The new namespace to associate with the dataset info.

        Returns:
            AbstractDatasetInfo: The new instance of AbstractDatasetInfo with the updated namespace.

        """

    @classmethod
    def extract_mapping_key_field(cls, dataset_mapping_key: str, dataset_mapping_fields: List[str]) -> Dict[str, str]:
        """
        Extracts the fields from a dataset mapping key and returns them as a dictionary.

        Args:
            dataset_mapping_key (str): The dataset mapping key to extract fields from.
            dataset_mapping_fields (List[str]): The list of fields to extract.

        Returns:
            Dict[str, str]: The extracted fields as a dictionary.

        Raises:
            ValueError: If the number of fields in the mapping key does not match the expected number of fields.

        """
        fields = dataset_mapping_key.split(cls.SEPARATOR)

        if len(fields) != len(dataset_mapping_fields):
            raise ValueError(f"Wrong format for {dataset_mapping_key}, "
                             f"the expected format "
                             f"is {cls.SEPARATOR.join(dataset_mapping_fields)}")

        else:
            return dict(zip(dataset_mapping_fields, fields))
