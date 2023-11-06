'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/31-2023/11/03 Copyright 2023 Capgemini

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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
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

from sostrades_core.execution_engine.data_connector.abstract_data_connector import (
    AbstractDataConnector,
)
import requests

GLOBAL_EXECUTION_ENGINE_ONTOLOGY_IDENTIFIER = 'GLOBAL_EXECUTION_ENGINE_ONTOLOGY'


class OntologyDataConnector(AbstractDataConnector):
    """
    Specific data connector for Ontology
    """

    data_connection_list = ['endpoint']

    NAME = 'ONTOLOGY'

    CONNECTOR_TYPE = 'connector_type'
    CONNECTOR_DATA = 'connector_data'

    REQUEST_TYPE = 'REQUEST_TYPE'
    REQUEST_ARGS = 'REQUEST_ARGS'
    PARAMETER_REQUEST = 'parameter'

    def __init__(self, data_connection_info=None):
        """
        Constructor for Ontology data connector

        :param data_connection_info: contains necessary data for connection
        :type data_connection_info: dict
        """
        self.endpoint = None
        self.route = None
        self.args = None

        super().__init__(data_connection_info=data_connection_info)

    def _extract_connection_info(self, data_connection_info):
        """
        Convert structure with data connection info given as parameter into member variable

        :param data_connection_info: contains necessary data for connection
        :type data_connection_info: dict
        """
        if 'endpoint' in data_connection_info:
            self.endpoint = data_connection_info['endpoint']

    def load_data(self, connection_data):
        """
        Method to load a data from ontology regarding input information

        :param: connection_data_dict, contains the necessary information to connect to Ontology API with request
        :type: dict

        """

        self._extract_connection_info(connection_data)

        result = {}

        if (
            connection_data[OntologyDataConnector.REQUEST_TYPE]
            == OntologyDataConnector.PARAMETER_REQUEST
        ):

            # Prepare default result
            for parameter_id in connection_data[OntologyDataConnector.REQUEST_ARGS]:
                result[parameter_id] = parameter_id

            # Prepare request payload
            payload = {
                'ontology_request': {
                    'parameters': connection_data[OntologyDataConnector.REQUEST_ARGS]
                }
            }
            # Launch request
            complete_url = f'{self.endpoint}'

            try:
                resp = requests.request(
                    method='POST', url=complete_url, json=payload, verify=False
                )

                if resp.status_code == 200:
                    ontology_response_data = resp.json()

                    if 'parameters' in ontology_response_data:
                        parameters_data = ontology_response_data['parameters']
                        for parameter_id in result.keys():
                            if parameter_id in parameters_data:
                                if 'unit' not in parameters_data[parameter_id]:
                                    parameters_data[parameter_id]['unit'] = ''
                                if 'label' in parameters_data[parameter_id]:
                                    result[parameter_id] = [
                                        parameters_data[parameter_id]['label'],
                                        parameters_data[parameter_id]['unit'],
                                    ]

            except Exception as ex:
                print(
                    'The following exception occurs when trying to reach Ontology server',
                    ex,
                )

            return result

    def write_data(self, connection_data):

        raise Exception("method not implemented")

    def set_connector_request(self, connector_info, request_type, request_args):
        """
        Update connector dictionary with request information
        :param connector_info: dictionary regarding connection information, must map TrinoDataConnector.data_connection_list
        :param request_type: type of request to do to Ontology
        :param args: attended entry to perform the request
        :return:
        """

        connector_info[OntologyDataConnector.REQUEST_TYPE] = request_type
        connector_info[OntologyDataConnector.REQUEST_ARGS] = request_args
        return connector_info


if __name__ == '__main__':

    ontology_connector = OntologyDataConnector()

    data_connection = {
        'endpoint': 'http://127.0.0.1:5555/api/ontology'
        # 'endpoint': 'http://127.0.0.1:5555/api/ontology'
    }

    args = [
        "CCS_price",
        "CO2_damage_price",
        "CO2_emissions_df",
        "CO2_emitted_forest_df",
        "CO2_objective",
        "CO2_taxes",
        "acceleration",
        "alpha",
        "authorize_self_coupled_disciplines",
        "beta",
        "cache_file_path",
        "cache_type",
        "carboncycle_df",
        "ccs_list",
        "chain_linearize",
        "conso_elasticity",
        "damage_df",
        "deforestation_surface",
        "economics_df",
        "energy_investment",
        "energy_list",
        "epsilon0",
        "forest_investment",
    ]

    ontology_connector.set_connector_request(
        data_connection, OntologyDataConnector.PARAMETER_REQUEST, args
    )

    result = ontology_connector.load_data(data_connection)

    print(result)
