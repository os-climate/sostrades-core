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

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class Disc5(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc5dict',
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
        'z': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'dict_out': {'type': 'dict', 'subtype_descriptor':{'dict':'float'}, 'default': {'key1': 1, 'key2': 2}, 'visibility': 'Shared', 'namespace': 'ns_test'},
        'dict_of_dict_out': {'type': 'dict', 'subtype_descriptor':{'dict':{'dict':'float'}}, 'default': {'key1': {'1': 1, '2': 2}}, 'visibility': 'Shared', 'namespace': 'ns_test'}
    }

    DESC_OUT = {
        'h': {'type': 'array', 'visibility': 'Shared', 'namespace': 'ns_test'}
    }

    def run(self):
        dict_out = self.get_sosdisc_inputs('dict_out')
        key1 = dict_out['key1']
        if isinstance(key1, dict):
            key1 = key1['1']
        key2 = dict_out['key2']
        if isinstance(key2, dict):
            key2 = key2['1']
        z = self.get_sosdisc_inputs('z')

        h = array([0.5 * (key1 + 1. / (2 * key1)),
                   0.5 * (key2 + 1. / (2 * key2))])

        dict_values = {'h': h}
        self.store_sos_outputs_values(dict_values)
