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
from __future__ import annotations

import logging
from typing import Any

from sostrades_core.datasets.dataset_info.dataset_info_versions import VERSION_V0, VERSION_V1
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import DatasetGenericException
from sostrades_core.datasets.datasets_connectors.abstract_multiversion_datasets_connector import (
    AbstractMultiVersionDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.\
    local_filesystem_datasets_connectorV0 import LocalFileSystemDatasetsConnectorV0
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.\
    local_filesystem_datasets_connectorV1 import LocalFileSystemDatasetsConnectorV1


class LocalFileSystemDatasetsConnectorMV(AbstractMultiVersionDatasetsConnector):
    """Specific multi-version dataset connector for datasets in local filesystem."""

    __logger = logging.getLogger(__name__)
    VERSION_TO_CLASS = {VERSION_V0: LocalFileSystemDatasetsConnectorV0,
                        VERSION_V1: LocalFileSystemDatasetsConnectorV1}
    ROOT_DIRECTORY_PATH_ARG = "root_directory_path"
    CREATE_IF_NOT_EXISTS_ARG = "create_if_not_exists"

    def __init__(self,
                 connector_id: str,
                 mono_version_connector_instantiation_fields: dict[str:dict[str:Any]]) -> None:
        """
        Multi-version constructor with the instantiation arguments of connectors of type LocalFileSystem. Note that the
        different mono-version sub-connectors should use different databases (root directories in the file system).

        Args:
            connector_id: Connector identifier for the multiversion LocalFileSystem connector
            mono_version_connector_instantiation_fields: keyword arguments that allow to instantiate each different
                mono-version LocalFileSystem connectors (cf. mono-version classes).

        """
        if len({_args[self.ROOT_DIRECTORY_PATH_ARG] for _args in mono_version_connector_instantiation_fields.values()}
               ) < len(mono_version_connector_instantiation_fields):
            raise DatasetGenericException(f"Not possible to instantiate a {self.__class__.__name__} that has two or "
                                          f"more mono-version sub-connectors with the same "
                                          f"{self.ROOT_DIRECTORY_PATH_ARG}")
        super().__init__(connector_id=connector_id,
                         mono_version_connectors_instantiation_fields=mono_version_connector_instantiation_fields)
