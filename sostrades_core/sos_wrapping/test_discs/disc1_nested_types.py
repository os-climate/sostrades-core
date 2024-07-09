'''
Copyright 2024 Capgemini

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

VARIABLES = "variable"
VALUES = "value"
UPPER_BOUND = "upper_bnd"
LOWER_BOUND = "lower_bnd"
TYPE = "type"
ENABLE_VARIABLE_BOOL = "enable_variable"
LIST_ACTIVATED_ELEM = "activated_elem"

class Disc1(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Disc1_nested_types',
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
    # TODO: subtype descriptors !?
    DESC_IN = {

        'X_dict_df': {'type': 'dict'},
        'X_dict_dict_df': {'type': 'dict'},
        'X_dict_dict_float': {'type': 'dict'},  # NB: this is perfectly jsonifiable no need to pkl
        'X_array_string': {'type': 'array'},
        'X_array_df': {'type': 'array'},
        'X_dspace_lists': {'type': 'dataframe', 'dataframe_descriptor': {VARIABLES: ('string', None, True),
                                                                         VALUES: ('multiple', None, True),
                                                                         LOWER_BOUND: ('multiple', None, True),
                                                                         UPPER_BOUND: ('multiple', None, True),
                                                                         ENABLE_VARIABLE_BOOL: ('bool', None, True),
                                                                         LIST_ACTIVATED_ELEM: ('list', None, True), }},
        'X_dspace_array': {'type': 'dataframe', 'dataframe_descriptor': {VARIABLES: ('string', None, True),
                                                                         VALUES: ('multiple', None, True),
                                                                         LOWER_BOUND: ('multiple', None, True),
                                                                         UPPER_BOUND: ('multiple', None, True),
                                                                         ENABLE_VARIABLE_BOOL: ('bool', None, True),
                                                                         LIST_ACTIVATED_ELEM: ('list', None, True), }},
    }

    def run(self):
        pass

