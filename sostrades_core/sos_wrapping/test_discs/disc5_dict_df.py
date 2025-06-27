'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
from numpy import array
from pandas import DataFrame

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class Disc5(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc5_dict_df',
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
        'z': {'type': 'array',  'namespace': 'ns_test'},
        'dict_out': {'type': 'dict',  'namespace': 'ns_test'}
    }

    DESC_OUT = {
        'h': {'type': 'dict', 'subtype_descriptor': {'dict': 'dataframe'},  'namespace': 'ns_test'}
    }

    def run(self):
        dict_out = self.get_sosdisc_inputs('dict_out')
        key1 = dict_out['key1']
        key11 = key1['key11']
        key14 = key1['key14']
        df = key14['key141']
        val = df['col2'][1]
        z = self.get_sosdisc_inputs('z')

        h_data = array([0.5 * (key11 + 1. / (2 * key11)),
                        0.5 * (val + 1. / (2 * val))])
        h = {'dataframe': DataFrame(data={'col1': h_data})}
        dict_values = {'h': h}
        self.store_sos_outputs_values(dict_values)
