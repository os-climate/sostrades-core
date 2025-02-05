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

import logging
from enum import Enum

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetUnableToInitializeConnectorException,
)
from sostrades_core.datasets.datasets_connectors.arango_datasets_connector import (
    ArangoDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.bigquery_datasets_connector import (
    BigqueryDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.json_datasets_connector.\
    json_datasets_connector_multiversion import (
    JSONDatasetsConnectorMV,
)
from sostrades_core.datasets.datasets_connectors.json_datasets_connector.json_datasets_connectorV0 import (
    JSONDatasetsConnectorV0,
)
from sostrades_core.datasets.datasets_connectors.json_datasets_connector.json_datasets_connectorV1 import (
    JSONDatasetsConnectorV1,
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.\
    local_filesystem_datasets_connector_multiversion import (
    LocalFileSystemDatasetsConnectorMV,
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.\
    local_filesystem_datasets_connectorV1 import (
    LocalFileSystemDatasetsConnectorV1,
)
from sostrades_core.datasets.datasets_connectors.local_filesystem_datasets_connector.local_filesystem_datasets_connectorV0 import (
    LocalFileSystemDatasetsConnectorV0,
)
from sostrades_core.datasets.datasets_connectors.local_repository_datasets_connector import (
    LocalRepositoryDatasetsConnector,
)
from sostrades_core.datasets.datasets_connectors.sospickle_datasets_connector import (
    SoSPickleDatasetsConnector,
)
from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta


class DatasetConnectorType(Enum):
    """
    Dataset connector types enum
    """
    JSON = JSONDatasetsConnectorV0
    JSON_V1 = JSONDatasetsConnectorV1
    JSON_MV = JSONDatasetsConnectorMV
    Local = LocalFileSystemDatasetsConnectorV0
    Local_V1 = LocalFileSystemDatasetsConnectorV1
    Local_MV = LocalFileSystemDatasetsConnectorMV
    Arango = ArangoDatasetsConnector
    SoSpickle = SoSPickleDatasetsConnector
    Local_repository = LocalRepositoryDatasetsConnector
    Bigquery = BigqueryDatasetsConnector

    @classmethod
    def get_enum_value(cls, value_str: str) -> DatasetConnectorType:
        """
        Get the enum value corresponding to the given string.

        Args:
            value_str (str): The string representation of the enum value.

        Returns:
            DatasetConnectorType: The corresponding enum value.

        Raises:
            DatasetUnableToInitializeConnectorException: If no matching enum value is found.
        """
        try:
            # Iterate through the enum members and find the one with a matching value
            return next(member for member in cls if member.name == value_str)
        except StopIteration:
            raise DatasetUnableToInitializeConnectorException(f"No matching enum value found for '{value_str}'")


class DatasetsConnectorFactory(metaclass=NoInstanceMeta):
    """
    Dataset connector factory
    """
    __logger = logging.getLogger(__name__)

    @classmethod
    def get_connector(cls, connector_identifier: str,
                      connector_type: DatasetConnectorType, **connector_instanciation_fields: dict) -> AbstractDatasetsConnector:
        """
        Instantiate a connector of type connector_type with provided arguments.
        Raises DatasetUnableToInitializeConnectorException if type is invalid.

        Args:
            connector_identifier (str): The identifier for the connector.
            connector_type (DatasetConnectorType): The type of connector to instantiate.
            **connector_instanciation_fields (dict): Additional fields for connector instantiation.

        Returns:
            AbstractDatasetsConnector: The instantiated connector.

        Raises:
            DatasetUnableToInitializeConnectorException: If the connector type is invalid.
        """
        cls.__logger.debug(f"Instantiating connector of type {connector_type}")
        if not isinstance(connector_type, DatasetConnectorType) or not issubclass(
            connector_type.value, AbstractDatasetsConnector
        ):
            raise DatasetUnableToInitializeConnectorException(f"Unexpected connector type {connector_type}")
        try:
            return connector_type.value(connector_identifier, **connector_instanciation_fields)
        except TypeError as exc:
            raise DatasetUnableToInitializeConnectorException(connector_type) from exc
