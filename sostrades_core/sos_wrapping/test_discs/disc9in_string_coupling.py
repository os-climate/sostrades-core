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
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class Disc9in(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc9in_string_coupling',
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
        'x': {'type': 'float'}}

    DESC_OUT = {
        'string': {'type': 'string', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'string_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'visibility': 'Shared',
                        'namespace': 'ns_test'},
        'string_dict': {'type': 'dict','subtype_descriptor': {'dict': 'string'}, 'visibility': 'Shared', 'namespace': 'ns_test'},
        'string_dict_of_dict': {'type': 'dict','subtype_descriptor': {'dict': {'dict':'string'}}, 'visibility': 'Shared', 'namespace': 'ns_test'},
        'dict_mix_types': {'type': 'dict', 'visibility': 'Shared', 'namespace': 'ns_test'},
        'dict_dict_dict_list_string': {'type': 'dict','subtype_descriptor': {'dict': {'dict':{'dict':{'list':'string'}}}}, 'visibility': 'Shared', 'namespace': 'ns_test'},
        'dict_list': {'type': 'list', 'subtype_descriptor': {'list': {'dict': {'dict': {'list': 'string'}}}},
                      'visibility': 'Shared', 'namespace': 'ns_test'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')

        if x > 0:
            dict_values = {'string': 'x is > 0',
                           'string_list': ['&1234(-_)=+6789$%!ABCabc', f'{x}{x}{x}{x}{x}{x}{x}', 'STEPS-HE'],
                           'string_dict': {'key0': 'STEPS-HE', 'key1': 'positive',
                                           'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'},
                           'string_dict_of_dict': {
                               'dict1': {'key1': 'STEPS-HE', 'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'},
                               'dict2': {'key1': 'positive', 'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'}},
                           'dict_mix_types': {'AC1': {'string': 'NA', 'float': x ** 3, 'integer': int(x ** (1.0 / 2.0)),
                                                      'list': [x, 2 * x], 'dict': {'key1': 'positive',
                                                                                   'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'}},
                                              'AC2': {'string': 'NA', 'float': x ** 4, 'integer': int(x ** (3.0 / 2.0)),
                                                      'list': [x, 3 * x], 'dict': {'key1': 'positive',
                                                                                   'key2': 'vvrnevqa81344U890aHDPI-----++++)))))____'}}},
                           'dict_list': [{'key_1': {
                               'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                               'key_2': {
                                   'scenario1': ['AC1', 'AC2']}}, {'key_11': {
                               'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                               'key_22': {
                                   'scenario1': ['AC1', 'AC2']}}],
                           'dict_dict_dict_list_string': {'s1': {'key_1': {
                               'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                               'key_2': {
                                   'scenario1': ['AC1', 'AC2']}},
                               's2': {'key_11': {
                                   'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                                   'key_22': {
                                       'scenario1': ['AC1', 'AC2']}}}
                           }
        else:
            dict_values = {'string': 'x is > 0',
                           'string_list': ['&1234(-_)=+6789$%!ABCabc', f'{x}{x}{x}{x}{x}{x}{x}', 'STEPS-HE'],
                           'string_dict': {},
                           'string_dict_of_dict': {},
                           'dict_mix_types': {},
                           'dict_list': [{'key_1': {
                               'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                               'key_2': {
                                   'scenario1': ['AC1', 'AC2']}}, {'key_11': {
                               'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                               'key_22': {
                                   'scenario1': ['AC1', 'AC2']}}],
                           'dict_dict_dict_list_string': {'s1': {'key_1': {
                               'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                               'key_2': {
                                   'scenario1': ['AC1', 'AC2']}},
                               's2': {'key_11': {
                                   'scenario1': ['AC1', 'AC2'], 'scenario2': ['AC3', 'AC4']},
                                   'key_22': {
                                       'scenario1': ['AC1', 'AC2']}}}
                           }
        self.store_sos_outputs_values(dict_values)
