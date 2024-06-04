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
import importlib
import json
import logging
from os.path import dirname, join
from shutil import rmtree
from typing import Any
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import DatasetGenericException
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector import LocalFileSystemDatasetsConnector


class LocalRepositoryDatasetsConnector(LocalFileSystemDatasetsConnector):
    """
    Local file system connector for default repository connector
    """
    DATASETS_DB_NAME = 'datasets_database'
    DATASETS_FOLDER_NAME = 'datasets'
    MAPPINGS_FOLDER_NAME = 'mappings'
    
    def __init__(self, module_name:str):
        """
        Constructor for Local Repository Datasets Connector

        
        :param module_name: name of the module root of the repository
        :type module_name: str
        
        """
        try:
            # import the module
            module = importlib.import_module(module_name)
        except Exception as exception:
            raise DatasetGenericException(f"Unable to import the module {module_name}: {exception}") from exception
        # find the module path
        root_path = dirname(module.__file__)
        # add the datasets default database to the path
        dataset_database_path = join(root_path, self.DATASETS_DB_NAME, self.DATASETS_FOLDER_NAME)
        super().__init__(root_directory_path=dataset_database_path, create_if_not_exists=False)
     
    
    @staticmethod
    def get_datasets_database_mappings_folder_path(module_name, file_name):
        """
            Method to find the datasets_database/mappings folder path for the given repository module name 
            ('sostrades_core' for example of the sostrades-core module name)
        """
        try:
            # import the module
            module = importlib.import_module(module_name)
        except Exception as exception:
            raise DatasetGenericException(f"Unable to import the module {module_name}: {exception}") from exception
        # find the module path
        root_path = dirname(module.__file__)
        # add the datasets default database to the path
        mapping_database_path = join(root_path, 
                                     LocalRepositoryDatasetsConnector.DATASETS_DB_NAME, 
                                     LocalRepositoryDatasetsConnector.MAPPINGS_FOLDER_NAME,
                                     file_name)
        return mapping_database_path