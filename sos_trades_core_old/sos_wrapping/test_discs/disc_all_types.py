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


class DiscAllTypes(SoSDiscipline):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc_all_types',
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
        'z': {'type': 'float', 'default': 90., 'unit': 'kg', 'user_level': 1,
              'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'h': {'type': 'array', 'unit': 'kg', 'user_level': 1},
        'dict_in': {'type': 'dict', SoSDiscipline.SUBTYPE: {'dict': 'float'}, 'unit': 'kg', 'user_level': 1},
        'df_in': {'type': 'dataframe', 'unit': 'kg', 'user_level': 1},
        'weather': {'type': 'string', 'default': 'cloudy, it is Toulouse ...', 'user_level': 1,
                    'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_of_dict_in': {'type': 'dict', SoSDiscipline.SUBTYPE: {'dict': {'dict': 'float'}}, 'user_level': 1},
        'dict_of_df_in': {'type': 'dict', SoSDiscipline.SUBTYPE: {'dict': 'dataframe'}, 'user_level': 1}
    }
    DESC_OUT = {
        'df_out': {'type': 'dataframe', 'unit': 'kg', 'user_level': 1,
                   'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'o': {'type': 'array', 'unit': 'kg', 'user_level': 1,
              'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_out': {'type': 'dict', 'unit': 'kg', 'user_level': 1,
                     'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'}
    }

    def run(self):
        h = self.get_sosdisc_inputs('h')
        z = self.get_sosdisc_inputs('z')
        dict_in = self.get_sosdisc_inputs('dict_in')
        df_in = self.get_sosdisc_inputs('df_in')
        dict_of_dict_in = self.get_sosdisc_inputs('dict_of_dict_in')
        z = z / 3.1416
        key1 = dict_in['key1'] + dict_of_dict_in['key_A']['subKey2']
        key2 = df_in['c2'][0] * dict_of_dict_in['key_B']['subKey1']
        h = array([0.5 * (h[0] + 1. / (2 * key1)),
                   0.5 * (h[-1] + 1. / (2 * key2))])
        dict_out = {'key1': ((h[0] + h[1]) / z * 100),
                    'key2': ((h[0] + h[1]) / z * 100)}
        dict_values = {'o': z,
                       'dict_out': dict_out}
        df_in = DataFrame(array([[(h[0] + h[1]) / 2, (h[0] + h[1]) / 2]]),
                          columns=['c1', 'c2'])
        df_in['z'] = [2 * z] * len(df_in)
        dict_values.update({'df_out': df_in})
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
