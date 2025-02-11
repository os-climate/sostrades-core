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
from os.path import dirname, join

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    DatasetGenericException,
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.local_filesystem_datasets_connectorV0 import (
    LocalFileSystemDatasetsConnectorV0,
)


class LocalRepositoryDatasetsConnector(LocalFileSystemDatasetsConnectorV0):
    """Local file system connector for default repository connector"""

    DATASETS_DB_NAME = 'datasets_database'
    DATASETS_FOLDER_NAME = 'datasets'
    MAPPINGS_FOLDER_NAME = 'mappings'

    def __init__(self, connector_id: str, module_name: str):
        """
        Constructor for Local Repository Datasets Connector

        Args:
            connector_id (str): An unique name to identify clearly this connector
            module_name (str): Name of the module root of the repository

        Raises:
            DatasetGenericException: If the module cannot be imported

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
        super().__init__(connector_id, root_directory_path=dataset_database_path, create_if_not_exists=False)

    @staticmethod
    def get_datasets_database_mappings_folder_path(module_name: str, file_name: str) -> str:
        """
        Method to find the datasets_database/mappings folder path for the given repository module name

        Args:
            module_name (str): Name of the module root of the repository
            file_name (str): Name of the file in the mappings folder

        Returns:
            str: The path to the mappings folder

        Raises:
            DatasetGenericException: If the module cannot be imported

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
