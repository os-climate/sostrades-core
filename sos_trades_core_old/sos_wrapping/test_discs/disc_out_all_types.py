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
# coding: utf-8
from numpy import array
from pandas import DataFrame

from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline


class DiscOutAllTypes(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc_out_all_types',
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
    DESC_IN = {
        'z': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'y': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'AC_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'year_start': {'type': 'int', 'default': 2050, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'year_end': {'type': 'int', 'default': 2050, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
    }
    DESC_OUT = {
        'df_out': {'type': 'dataframe',
                   'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_out': {'type': 'dict',
                     'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_dict_out': {'type': 'dict',
                          'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_df_out': {'type': 'dict', 'unit': 'kg', 'user_level': 1,
                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'}
    }

    def run(self):
        z = self.get_sosdisc_inputs('z')
        y = self.get_sosdisc_inputs('y')
        ac_list = self.get_sosdisc_inputs('AC_list')
        year_end = self.get_sosdisc_inputs('year_end')
        dict_out = {'A1':  z * 100,
                    'A2': y / 100}
        dict_out2 = {'years': [year_end],
                     'key1':  [z * 100],
                     'key2': [y / 100]}
        df_out = DataFrame(dict_out2)

        dict_df_out = {ac: df_out for ac in ac_list}
        dict_dict_out = {ac: dict_out for ac in ac_list}
        dict_values = {'dict_out': dict_out, 'df_out': df_out,
                       'dict_dict_out': dict_dict_out, 'dict_df_out': dict_df_out}

        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
