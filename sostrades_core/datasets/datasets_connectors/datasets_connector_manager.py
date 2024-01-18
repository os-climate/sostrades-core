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

from sostrades_core.datasets.datasets_connectors.abstract_datasets_connector import AbstractDatasetsConnector
from sostrades_core.datasets.datasets_connectors.datasets_connector_factory import DatasetsConnectorFactory, DatasetConnectorType
from sostrades_core.tools.metaclasses.no_instance import NoInstanceMeta


class DatasetsConnectorManager(metaclass=NoInstanceMeta):
    """
    Datasets connector manager
    """
    __registered_connectors = {}
    __logger = logging.getLogger(__name__)
    
    @classmethod
    def get_connector(cls, connector_identifier:str) -> AbstractDatasetsConnector:
        """
        Gets a connector given its identifier

        :param connector_identifier: identifier of the connector
        :type connector_identifier: string
        """
        cls.__logger.debug(f"Getting connector {connector_identifier}")
        if connector_identifier not in cls.__registered_connectors:
            raise ValueError(f"Connector {connector_identifier} not found.")
        return cls.__registered_connectors[connector_identifier]

    @classmethod
    def register_connector(cls, connector_identifier:str, connector_type:DatasetConnectorType, **connector_instanciation_fields) -> AbstractDatasetsConnector:
        """
        Register a connector with connector_instanciation_fields
        :param connector_identifier: An unique name to identify clearly this connector
        :type connector_identifier: str
        :param connector_type: Name of an existing connector
        :type connector_type: DatasetConnectorTypes
        :param connector_instanciation_fields: dictionary of key/value needed by the connector
        :type connector_instanciation_fields: dict
        """
        cls.__logger.debug(f"Registering connector {connector_identifier}")
        if connector_identifier in cls.__registered_connectors.keys():
            cls.__logger.debug(f'Existing connector \"{connector_identifier}\" is updated')

        connector = DatasetsConnectorFactory.get_connector(connector_type=connector_type, **connector_instanciation_fields)
        cls.__registered_connectors[connector_identifier] = connector 
        return connector
    