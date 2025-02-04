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
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.local_filesystem_datasets_connector_multiversion import (
    LocalFileSystemDatasetsConnectorMV,
)
from sostrades_core.datasets.dataset_info.dataset_info_versions import (VERSION_V0, VERSION_V1)

class LocalRepositoryDatasetsConnector(LocalFileSystemDatasetsConnectorMV):
    """
    Local file system connector for default repository connector
    """
    DATASETS_DB_NAME = 'datasets_database'
    DATASETS_FOLDER_NAME_V0 = 'datasets'
    DATASETS_FOLDER_NAME_V1 = 'datasets_V1'
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
        dataset_database_path_v0 = join(root_path, self.DATASETS_DB_NAME, self.DATASETS_FOLDER_NAME_V0)
        dataset_database_path_v1 = join(root_path, self.DATASETS_DB_NAME, self.DATASETS_FOLDER_NAME_V1)
        super().__init__(connector_id, mono_version_connector_instantiation_fields={
            VERSION_V0: {self.ROOT_DIRECTORY_PATH_ARG: dataset_database_path_v0,
                         self.CREATE_IF_NOT_EXISTS_ARG: False},
            VERSION_V1: {self.ROOT_DIRECTORY_PATH_ARG: dataset_database_path_v1,
                         self.CREATE_IF_NOT_EXISTS_ARG: True}
        })

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
