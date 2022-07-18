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
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
from sostrades_core.execution_engine.data_connector.dremio_data_connector import DremioDataConnector


class Disc1_data_connector_dremio(ProxyDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc1_data_connector_dremio',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    _maturity = 'Fake'

    hostname = 'fr0-idlabi-bw37.eu.airbus.corp'
    port = 31020
    username = 'sostrades'
    password = 'sosdrem1o'
    dremio_path = '"pyGMF"."GMF_VIEWS"."GMF_20_VINTAGE_2"."BY_SCENARIO_ID"."55 - sos_testing_scenario_55"."FCST_AOT_GDF_AGG"'

    data_connection_dict = {'connector_type': DremioDataConnector.NAME,
                            'hostname': hostname,
                            'port': port,
                            'username': username,
                            'password': password,
                            'connector_request': dremio_path}

    DESC_IN = {
        'x': {'type': 'float', 'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'a': {'type': 'float'},
        'b': {'type': 'float'},
        'y': {'type': 'dataframe', 'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac',
              ProxyDiscipline.CONNECTOR_DATA: data_connection_dict}
    }
    DESC_OUT = {
        'indicator': {'type': 'float'},

    }

    def run(self):
        x = self.get_sosdisc_inputs('x')
        a = self.get_sosdisc_inputs('a')
        b = self.get_sosdisc_inputs('b')
        # dict_values = {'indicator': a * b, 'y': a * x + b}
        dict_values = {'indicator': a * b}
        # put new field value in data_out

        # # get a connector_data with all information
        # connector_data = ConnectorFactory.set_connector_request(
        #     self.DESC_OUT['y'][SoSDiscipline.CONNECTOR_DATA], self.dremio_path)
        # # update the meta_data with the new connection information
        # # { var_name : {meta_dat_to_update : meta_data_value}}
        # self.update_meta_data_out(
        #     {'y': {SoSDiscipline.CONNECTOR_DATA: connector_data}})
        self.store_sos_outputs_values(dict_values)
