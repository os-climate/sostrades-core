'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/12-2023/11/03 Copyright 2023 Capgemini

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

from sostrades_core.execution_engine.data_connector.dremio_data_connector import (
    DremioDataConnector,
)
from sostrades_core.execution_engine.data_connector.trino_data_connector import (
    TrinoDataConnector,
)
from sostrades_core.execution_engine.data_connector.mock_connector import MockConnector
from sostrades_core.execution_engine.data_connector.ontology_data_connector import (
    OntologyDataConnector,
)
from sostrades_core.execution_engine.data_connector.mongodb_data_connector import MongoDBDataConnector


class ConnectorFactory:
    """
    Data connector factory
    """

    CONNECTOR_TYPE = 'connector_type'

    _CONNECTORS = {
        DremioDataConnector.NAME: DremioDataConnector,
        MockConnector.NAME: MockConnector,
        TrinoDataConnector.NAME: TrinoDataConnector,
        OntologyDataConnector.NAME: OntologyDataConnector,
        MongoDBDataConnector.NAME: MongoDBDataConnector
    }

    @staticmethod
    def set_connector_request(connector_info, request):

        if ConnectorFactory.CONNECTOR_TYPE in connector_info:
            connector_instance = ConnectorFactory._CONNECTORS[
                connector_info[ConnectorFactory.CONNECTOR_TYPE]
            ]()
            connector_instance.set_connector_request(connector_info, request)

        else:
            raise TypeError(f'Connector type not found in {connector_info}')

        return connector_info

    @staticmethod
    def use_data_connector(connector_info, logger=None):
        """
        create and instance of the required data connector

        :params: connector_info, information with for connection and request
        :type: dict
        """
        new_connector = None
        if ConnectorFactory.CONNECTOR_TYPE in connector_info:
            new_connector = ConnectorFactory._CONNECTORS[
                connector_info[ConnectorFactory.CONNECTOR_TYPE]
            ]()
        else:
            raise TypeError(f'Connector type not found in {connector_info}')
        try:
            if (
                new_connector.get_connector_mode(connector_info)
                == new_connector.CONNECTOR_MODE_READ
            ):
                data = new_connector.load_data(connector_info)
            else:
                data = new_connector.write_data(connector_info)
        except Exception as exp:
            str_error = f'Error while using data connector {connector_info[ConnectorFactory.CONNECTOR_TYPE]}: {str(exp)}'
            if logger is not None:
                logger.error(str_error)
            raise Exception(str_error)
        return data

    @staticmethod
    def get_connector(connector_type, connector_connexion_info):
        """
        Build and return a connector regarding given type
        :param connector_type: Name of an existing connector
        :type connector_type: str
        :param connector_connexion_info: dictionary of key/value needed by the connector
        :type connector_connexion_info: dict
        """

        connector = ConnectorFactory._CONNECTORS.get(connector_type)
        if not connector:
            raise TypeError(f'Connector type {connector_type} does not exist.')
        return connector(data_connection_info=connector_connexion_info)

    @staticmethod
    def get_connector_connexion_info(connector_type):
        """
        Build and return a connector regarding given type
        :param connector_type: Name of an existing connector
        :type connector_type: str
        """

        connector = ConnectorFactory._CONNECTORS.get(format)
        if not connector:
            raise TypeError(f'Connector type {connector_type} does not exist.')
        return connector.data_connection_list.copy()


class PersistentConnectorContainer:
    def __init__(self, logger: logging.Logger):
        """
        Class constructor
        """
        self.__registered_connectors = {}
        self.__logger = logger

    def register_persistent_connector(
        self, connector_type, connector_identifier, connector_connexion_info
    ):
        """
        Register a connector with connection info
        :param connector_type: Name of an existing connector
        :type connector_type: str
        :param connector_identifier: An unique name to identify clearly this connector
        :type connector_identifier: str
        :param connector_connexion_info: dictionary of key/value needed by the connector
        :type connector_connexion_info: dict
        """

        if connector_identifier in self.__registered_connectors.keys():
            self.__logger.info(f'Existing connector "{connector_identifier}" is updated')

        connector = ConnectorFactory.get_connector(connector_type, connector_connexion_info)
        self.__registered_connectors[
            connector_identifier
        ] = connector 
        return connector 

    def get_persistent_connector(self, connector_identifier):
        """
        Retrieve a connector previously registered
        :param connector_identifier: An unique name to identify clearly this connector
        :type connector_identifier: str

        :return: sostrades_core.execution_engine.data_connector.abstract_data_connector.AbstractDataConnector inherited instance
        """

        if connector_identifier not in self.__registered_connectors.keys():
            self.__logger.info(f'Request a non registered connector "{connector_identifier}"')
        return self.__registered_connectors.get(connector_identifier, None)
