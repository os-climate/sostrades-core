'''
Copyright 2022 Airbus SAS

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

from sos_trades_core.execution_engine.data_connector.dremio_data_connector import DremioDataConnector
from sos_trades_core.execution_engine.data_connector.trino_data_connector import TrinoDataConnector
from sos_trades_core.execution_engine.data_connector.mock_connector import MockConnector


class ConnectorFactory:
    """
    Data connector factory
    """

    CONNECTOR_TYPE = 'connector_type'

    CONNECTORS = {
        DremioDataConnector.NAME: DremioDataConnector,
        MockConnector.NAME: MockConnector,
        TrinoDataConnector.NAME: TrinoDataConnector
    }

    @staticmethod
    def set_connector_request(connector_info, request):

        if ConnectorFactory.CONNECTOR_TYPE in connector_info:
            connector_instance = ConnectorFactory.CONNECTORS[connector_info[ConnectorFactory.CONNECTOR_TYPE]](
            )
            connector_instance.set_connector_request(connector_info, request)

        else:
            raise TypeError(f'Connector type not found in {connector_info}')

        return connector_info

    """
    @staticmethod
    def get_connector(connector_identifier):
        return ConnectorFactory.CONNECTORS[connector_identifier]()
    """

    @staticmethod
    def use_data_connector(connector_info, logger=None):
        """
        create and instance of the required data connector

        :params: connector_info, information with for connection and request
        :type: dict
        """
        new_connector = None
        if ConnectorFactory.CONNECTOR_TYPE in connector_info:
            new_connector = ConnectorFactory.CONNECTORS[connector_info[ConnectorFactory.CONNECTOR_TYPE]](
            )
        else:
            raise TypeError(f'Connector type not found in {connector_info}')
        try:
            if new_connector.get_connector_mode(connector_info) == new_connector.CONNECTOR_MODE_READ:
                data = new_connector.load_data(connector_info)
            else:
                data = new_connector.write_data(connector_info)
        except Exception as exp:
            str_error = f'Error while using data connector {connector_info[ConnectorFactory.CONNECTOR_TYPE]}: {str(exp)}'
            if logger is not None:
                logger.error(str_error)
            raise Exception(str_error)
        return data
