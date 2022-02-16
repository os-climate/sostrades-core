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
from numpy import array


class Disc9out(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc9out_string_coupling',
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
        'string': {'type': 'string', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'string_list': {'type': 'string_list', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'string_dict': {'type': 'dict', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'string_dict_of_dict': {'type': 'dict', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'dict_mix_types': {'type': 'dict', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'dict_dict_dict_list_string': {'type': 'dict', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'dict_list': {'type': 'float_list', 'visibility': 'Shared', 'namespace': 'ns_test'}
    }

    DESC_OUT = {'z': {'type': 'float'}}

    def run(self):
        inputs = self.get_sosdisc_inputs()

        if inputs['string'] == 'x is > 0' and inputs['string_dict_of_dict']['dict2']['key1'] == 'positive':

            z = inputs['dict_mix_types']['AC2']['list'][0]

        self.store_sos_outputs_values({'z': z})
