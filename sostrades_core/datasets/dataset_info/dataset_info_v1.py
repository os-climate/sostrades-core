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
from sostrades_core.datasets.dataset_info.dataset_info_versions import VERSION_V1


@dataclass(frozen=True)
class DatasetInfoV1(AbstractDatasetInfo):
    """
    Stores the informations of a dataset V0
    """
    GROUP_ID_KEY = "group_id"
    MAPPING_ITEM_FIELDS = [
                        AbstractDatasetInfo.VERSION_ID_KEY,
                        AbstractDatasetInfo.CONNECTOR_ID_KEY,
                        AbstractDatasetInfo.DATASET_ID_KEY,
                        GROUP_ID_KEY,
                        AbstractDatasetInfo.PARAMETER_ID_KEY
                        ]
    group_id:str

    @property
    def dataset_info_id(self) -> str:
        return self.get_mapping_id([self.version_id, self.connector_id, self.dataset_id, self.group_id])


    @property
    def version_id(self) -> str:
        return VERSION_V1

    @staticmethod
    def deserialize(dataset_mapping_key:str) -> dict[str,str]:
        """
        Method to deserialize
        expected
        'vx(optional)|'<connector_id>|<dataset_id>|<group_id>|<parameter_id> (for V1)
        :param dataset_mapping_key: datasets informations of mapping dataset
        :type dataset_mapping_key: str
        """
        # check if the version is in the mapping key or not
        input_dict = DatasetInfoV1.extract_mapping_key_field(dataset_mapping_key, DatasetInfoV1.MAPPING_ITEM_FIELDS)
        if input_dict[DatasetInfoV1.DATASET_ID_KEY] == DatasetInfoV1.WILDCARD:
            raise DatasetsInfoMappingException(f"Wrong format for V1 mapping key {dataset_mapping_key}, the dataset name '*' is not authorised")
        return input_dict


    @staticmethod
    def create(input_dict: dict) -> DatasetInfoV1:

        return DatasetInfoV1(
            connector_id=input_dict[DatasetInfoV1.CONNECTOR_ID_KEY],
            dataset_id=input_dict[DatasetInfoV1.DATASET_ID_KEY],
            group_id=input_dict[DatasetInfoV1.GROUP_ID_KEY]
        )

    def copy_with_new_ns(self, associated_namespace:str)-> AbstractDatasetInfo:
        '''
        overrided method
        '''
        group_id = ""
        # if the group id is wildcard, the copy will have the group id as namespace
        if self.group_id == self.WILDCARD:
            group_id = associated_namespace
        else:
            group_id = self.group_id

        return DatasetInfoV1(
            connector_id=self.connector_id,
            dataset_id=self.dataset_id,
            group_id=group_id
        )

