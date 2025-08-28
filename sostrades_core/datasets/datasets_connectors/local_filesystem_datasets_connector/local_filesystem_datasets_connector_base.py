'''
Copyright 2025 Capgemini

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
import abc
import os

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
)
from sostrades_core.datasets.datasets_serializers.filesystem_datasets_serializer import (
    FileSystemDatasetsSerializer,
)
from sostrades_core.tools.folder_operations import makedirs_safe, rmtree_safe


class LocalFileSystemDatasetsConnectorBase(AbstractDatasetsConnector, abc.ABC):
    """Specific dataset connector for dataset in local filesystem"""

    DESCRIPTOR_FILE_NAME = 'descriptor.json'

    def __init__(self, connector_id: str, root_directory_path: str,
                 create_if_not_exists: bool) -> None:
        """
        Constructor for Local Filesystem data connector

        Args:
            connector_id (str): The identifier for the connector.
            root_directory_path (str): Root directory path for this dataset connector using filesystem.
            create_if_not_exists (bool, optional): Whether to create the root directory if it does not exist. Defaults to False.

        """
        super().__init__()
        self._root_directory_path = os.path.abspath(root_directory_path)
        self._create_if_not_exists = create_if_not_exists

        # create dataset folder if it does not exists
        if self._create_if_not_exists and not os.path.isdir(self._root_directory_path):
            makedirs_safe(self._root_directory_path, exist_ok=True)

        # configure dataset serializer
        self._datasets_serializer = FileSystemDatasetsSerializer()

        self.connector_id = connector_id

    def clear_connector(self) -> None:
        """Removes the entire root directory of the FileSystem connector and all datasets in it."""
        rmtree_safe(self._root_directory_path)

    def clear_dataset(self, dataset_id: str) -> None:
        """
        Utility method to remove the directory corresponding to a given dataset_id within the root directory.

        Args:
            dataset_id (str): Identifier of the dataset to be removed.

        """
        dataset_pth = os.path.join(self._root_directory_path, dataset_id)
        if os.path.exists(dataset_pth):
            rmtree_safe(dataset_pth)
