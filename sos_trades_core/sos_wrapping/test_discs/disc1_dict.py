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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from pandas import DataFrame


class Disc1(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc1_dict',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'x_list': {'type': 'float_list', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'a': {'type': 'float'},
        'b': {'type': 'float'}
    }
    DESC_OUT = {
        'indicator': {'type': 'float'},
        'y_dict': {'type': 'dict', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        't_dict': {'type': 'dict', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x_list')
        a = self.get_sosdisc_inputs('a')
        b = self.get_sosdisc_inputs('b')

        y_dict = {}
        t_dict = {}
        for i in range(len(x)):
            y_dict['name_' + str(i + 1)] = a * x[i] + b
            t_dict['name_' + str(i + 1)] = a + i + 1
        y_df = DataFrame.from_dict(y_dict, orient='index', columns=['y'])
        y_df.reset_index(level=0, inplace=True)
        dict_values = {'indicator': a * b,
                       'y_dict': y_dict,
                       't_dict': t_dict}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
