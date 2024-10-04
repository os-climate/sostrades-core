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

class DatasetsInfoMappingException(Exception):
    """
    Generic exception for dataset info
    """
    pass

@dataclass(frozen=True)
class AbstractDatasetInfo(abc.ABC):
    """
    Stores the informations of a dataset
    """
    # Keys for parsing json
    VERSION_ID_KEY = "version_id"
    CONNECTOR_ID_KEY = "connector_id"
    DATASET_ID_KEY = "dataset_id"
    PARAMETER_ID_KEY = "parameter_name"

    WILDCARD = "*"
    SEPARATOR = '|'
    
    # Id of the connector
    connector_id: str
    # Dataset id for this connector
    dataset_id: str

    @property
    @abc.abstractmethod
    def version_id(self)-> str:
        '''
        version to be override in each subclass
        '''

    @property
    def dataset_info_id(self) -> str:
        return self.get_mapping_id([self.version_id, self.connector_id, self.dataset_id])

    @staticmethod
    def get_mapping_id(ids: list[str]) -> str:
        return AbstractDatasetInfo.SEPARATOR.join(ids)

    
    @staticmethod
    @abc.abstractmethod
    def deserialize(dataset_mapping_key:str) -> dict[str:str]:
        """
        Method to deserialize
        expected
        <connector_id>|<dataset_id>|<parameter_id> (for V0)
        
        :param dataset_mapping_key: datasets informations of mapping dataset
        :type dataset_mapping_key: str
        """

    @staticmethod
    @abc.abstractmethod
    def create(input_dict:dict[str:str]) -> AbstractDatasetInfo:
        """
        Method to create the instance of datasetInfo
        expected
        {
        "version_id":<version_id>,
        "connector_id": <connector_id>,
        "dataset_id": <dataset_id>
        ...
         } (for V0)
        
        :param input_dict: datasets informations of mapping dataset
        :type input_dict: dict
        """

    @abc.abstractmethod
    def copy_with_new_ns(self, associated_namespace:str)-> AbstractDatasetInfo:
        '''
        create a new DatasetInfo instance from self
        Check if there is wilcard in the dataset info and update it with namespace info if needed
        '''

    

    



    