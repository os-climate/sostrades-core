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

from sostrades_core.tools.import_tool.import_tool import get_class_from_path
from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import (
    AbstractDatasetsConnector,
    DatasetUnableToInitializeConnectorException,
)
from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta


class DatasetsConnectorFactory(metaclass=NoInstanceMeta):
    """
    Dataset connector factory
    """
    __logger = logging.getLogger(__name__)

    @classmethod
    def get_connector(cls, connector_identifier: str,
                      connector_type: str,
                      **connector_instanciation_fields: dict) -> AbstractDatasetsConnector:
        """
        Instantiate a connector of type connector_type with provided arguments.
        Raises DatasetUnableToInitializeConnectorException if type is invalid.

        Args:
            connector_identifier (str): The identifier for the connector.
            connector_type (str): The type of connector to instantiate.
            **connector_instanciation_fields (dict): Additional fields for connector instantiation.

        Returns:
            AbstractDatasetsConnector: The instantiated connector.

        Raises:
            DatasetUnableToInitializeConnectorException: If the connector type is invalid.
        """
        cls.__logger.debug(f"Instantiating connector of type {connector_type}")

        try:
            connector_cls = get_class_from_path(connector_type)
        except Exception as exc: # TODO: ImportError?
            raise DatasetUnableToInitializeConnectorException(connector_type) from exc

        if  not issubclass(connector_cls, AbstractDatasetsConnector):
            raise DatasetUnableToInitializeConnectorException(f"Unexpected connector type {connector_type}")
        try:
            return connector_cls(connector_identifier, **connector_instanciation_fields)
        except TypeError as exc:
            raise DatasetUnableToInitializeConnectorException(connector_type) from exc
