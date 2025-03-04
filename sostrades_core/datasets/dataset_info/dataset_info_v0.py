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

from dataclasses import dataclass

from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo, DatasetsInfoMappingException
from sostrades_core.datasets.dataset_info.dataset_info_versions import VERSION_V0


@dataclass(frozen=True)
class DatasetInfoV0(AbstractDatasetInfo):
    """Stores the informations of a dataset V0"""

    MAPPING_ITEM_FIELDS = [
                        AbstractDatasetInfo.VERSION_ID_KEY,
                        AbstractDatasetInfo.CONNECTOR_ID_KEY,
                        AbstractDatasetInfo.DATASET_ID_KEY,
                        AbstractDatasetInfo.PARAMETER_ID_KEY
                        ]

    @property
    def version_id(self) -> str:
        """
        Returns the version ID of the dataset.

        Returns:
            str: The version ID of the dataset.

        """
        return VERSION_V0

    @staticmethod
    def deserialize(dataset_mapping_key: str) -> dict[str, str]:
        """
        Method to deserialize
        expected
        'vx(optional)|'<connector_id>|<dataset_id>|<parameter_id> (for V0)

        Args:
            dataset_mapping_key (str): datasets informations of mapping dataset

        Returns:
            dict[str, str]: A dictionary containing the deserialized dataset information.

        Raises:
            DatasetsInfoMappingException: If the dataset name is '*'.

        """
        # check if the version is in the mapping key or not
        input_dict = DatasetInfoV0.extract_mapping_key_field(dataset_mapping_key, DatasetInfoV0.MAPPING_ITEM_FIELDS)
        if input_dict[DatasetInfoV0.DATASET_ID_KEY] == DatasetInfoV0.WILDCARD:
            raise DatasetsInfoMappingException(f"Wrong format for V0 mapping key {dataset_mapping_key}, the dataset name '*' is not authorised")
        return input_dict

    @staticmethod
    def create(input_dict: dict[str, str]) -> DatasetInfoV0:
        """
        Method to create a DatasetInfoV0 instance.

        Args:
            input_dict (dict[str, str]): A dictionary containing the dataset information.

        Returns:
            DatasetInfoV0: An instance of DatasetInfoV0.

        """
        return DatasetInfoV0(
            connector_id=input_dict[DatasetInfoV0.CONNECTOR_ID_KEY],
            dataset_id=input_dict[DatasetInfoV0.DATASET_ID_KEY]
        )

    def copy_with_new_ns(self, associated_namespace: str) -> AbstractDatasetInfo:
        """
        Overrided method to create a copy of the dataset info with a new namespace.

        Args:
            associated_namespace (str): The new namespace to associate with the dataset info.

        Returns:
            AbstractDatasetInfo: A new instance of DatasetInfoV0 with the updated namespace.

        """
        # there is no need to update ns in this dataset info version
        return DatasetInfoV0(
            connector_id=self.connector_id,
            dataset_id=self.dataset_id
        )

    @classmethod
    def extract_mapping_key_field(cls, dataset_mapping_key: str, dataset_mapping_fields: list[str]) -> dict[str, str]:
        """
        Extracts the fields from the dataset mapping key.

        Args:
            dataset_mapping_key (str): The dataset mapping key.
            dataset_mapping_fields (list[str]): The list of dataset mapping fields.

        Returns:
            dict[str, str]: A dictionary containing the extracted fields.

        Raises:
            ValueError: If the format of the dataset mapping key is incorrect.

        """
        # check if there is a version in the key, if not it is V0
        if not (dataset_mapping_key.startswith(f'V0{cls.SEPARATOR}') or dataset_mapping_key.startswith(f'v0{cls.SEPARATOR}')):
            dataset_mapping_key = f'{VERSION_V0}{cls.SEPARATOR}{dataset_mapping_key}'

        fields = dataset_mapping_key.split(cls.SEPARATOR)

        if len(fields) != len(dataset_mapping_fields):
            raise ValueError(f"Wrong format for {dataset_mapping_key}, "
                             f"the expected format "
                             f"is {cls.SEPARATOR.join(dataset_mapping_fields)}")

        else:
            return dict(zip(dataset_mapping_fields, fields))
