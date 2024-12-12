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
import logging
import os.path
from sostrades_core.tools.folder_operations import rmtree_safe
from typing import Any

from sostrades_core.datasets.dataset_info.dataset_info_versions import (VERSION_V0, VERSION_V1)
from sostrades_core.datasets.dataset_info.abstract_dataset_info import AbstractDatasetInfo
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    DatasetGenericException
)
from sostrades_core.datasets.datasets_connectors.abstract_multiversion_datasets_connector import (
    AbstractMultiVersionDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.\
    local_filesystem_datasets_connectorV0 import LocalFileSystemDatasetsConnectorV0
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.\
    local_filesystem_datasets_connectorV1 import LocalFileSystemDatasetsConnectorV1
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.\
    local_filesystem_datasets_connector_base import LocalFileSystemDatasetsConnectorBase
from sostrades_core.datasets.datasets_serializers.datasets_serializer_factory import (
    DatasetSerializerType,
)


class LocalFileSystemDatasetsConnectorMV(AbstractMultiVersionDatasetsConnector):  # FIXME: remove the MV when all is tested
    """
    Specific multi-version dataset connector for datasets in local filesystem.
    """
    __logger = logging.getLogger(__name__)
    VERSION_TO_CLASS = {VERSION_V0: LocalFileSystemDatasetsConnectorV0,
                        VERSION_V1: LocalFileSystemDatasetsConnectorV1}
    ROOT_DIRECTORY_PATH_ARG = "root_directory_path"

    def __init__(self,
                 connector_id: str,
                 mono_version_connector_instantiation_fields: dict[str:dict[str:Any]]):
        """
        Multi-version constructor with the instantiation arguments of connectors of type Local (FileSystem).

        Args:
            connector_id (str): The identifier for the connector.
            root_directory_path (str): Root directory path for this dataset connector using filesystem.
            create_if_not_exists (bool, optional): Whether to create the root directory if it does not exist. Defaults
                to False.
            serializer_type (DatasetSerializerType, optional): Type of serializer to deserialize data from connector.
                Defaults to DatasetSerializerType.FileSystem.
        """
        if len({_args[self.ROOT_DIRECTORY_PATH_ARG] for _args in mono_version_connector_instantiation_fields.values()}
               ) < len(mono_version_connector_instantiation_fields):
            raise DatasetGenericException(f"Not possible to instantiate a {self.__class__.__name__} that has two or "
                                          f"more mono-version sub-connectors with the same "
                                          f"{self.ROOT_DIRECTORY_PATH_ARG}")
        super().__init__(connector_id=connector_id,
                         mono_version_connectors_instantiation_fields=mono_version_connector_instantiation_fields)


    # TODO [discuss] double inheritance or code duplication? :D
    # def clear_connector(self):
    #     rmtree_safe(self._root_directory_path)
    #
    # def clear_dataset(self, dataset_id: str) -> None:
    #     """
    #     Utility method to remove the directory corresponding to a given dataset_id within the root directory.
    #
    #     Args:
    #         dataset_id (str): Identifier of the dataset to be removed.
    #     """
    #     rmtree_safe(os.path.join(self._root_directory_path, dataset_id))

