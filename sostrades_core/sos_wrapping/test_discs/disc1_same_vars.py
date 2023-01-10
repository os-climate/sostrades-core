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
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
import pandas as pd
import numpy as np


class Disc1(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Disc10 discipline with dynamic inputs',
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
    DESC_IN = {('a', 'local'): {'type': 'float'},
               'a': {'type': 'float'},
               ('a', 'local2'): {'type': 'float'},

               }
    DESC_OUT = {
        'y': {'type': 'float'},
        'true_x': {'type': 'float'},
        'test_a': {'type': 'bool'}
    }
    # ns_b is a new namespace defined for a dynamic variable in setup_sos_disciplines
    DYNAMIC_VAR_NAMESPACE_LIST = ['ns_x1', 'ns_x2']

    def setup_sos_disciplines(self):
        dynamic_inputs = {}

        dynamic_inputs.update({('x', 'ns_x1'): {'type': 'float',
                                                'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                'namespace': 'ns_x1'},
                               ('x', 'ns_x2'): {'type': 'float',
                                                'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                'namespace': 'ns_x2'},
                               ('x', 'true_x'): {'type': 'float',
                                                 'visibility': SoSWrapp.SHARED_VISIBILITY,
                                                 'namespace': 'ns_x'}}
                              )

        self.add_inputs(dynamic_inputs)

    def run(self):
        input_dict = self.get_sosdisc_inputs()
        a_from_tuple = input_dict[('a', 'local')]

        a = input_dict['a']
        x1 = input_dict[('x', 'ns_x1')]
        x2 = input_dict[('x', 'ns_x2')]
        true_x = input_dict[('x', 'true_x')]
        y = x1 + x2
        dict_values = {}
        dict_values['y'] = y
        dict_values['true_x'] = true_x
        dict_values['test_a'] = a == a_from_tuple
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
