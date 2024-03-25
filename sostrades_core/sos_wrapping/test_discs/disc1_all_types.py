'''
Copyright 2022 Airbus SAS
Modifications on 2023/09/22-2023/11/03 Copyright 2023 Capgemini

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
import numpy as np
import pandas as pd

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

class Disc1(SoSWrapp):
    # ontology information
    _ontology_data = {
        'label': 'Disc1_all_types',
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
        'x': {'type': 'float', 'range': [1., 10.]},
        'a': {'type': 'int'},
        'b': {'type': 'int', 'possible_values': [0, 2, 5]},
        'name': {'type': 'string', 'possible_values': ['A1', 'A2', 'A3']},
        'x_dict': {'type': 'dict', 'default': {}},

        'y_array': {'type': 'array', 'default': np.array([])},
        'z_list': {'type': 'list', 'default': []},
        'b_bool': {'type': 'bool', 'default':True},
        'd': {'type': 'dataframe', 
              'dataframe_descriptor': {"years":('int',None,True),"x":('float',None,True)},
              'default':pd.DataFrame(columns=["years", "x"])}
    }
    DESC_OUT = {
        'indicator': {'type': 'int'},
        'y': {'type': 'float'},
        'y_dict': {'type': 'dict'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')
        a = self.get_sosdisc_inputs('a')
        b = self.get_sosdisc_inputs('b')
        name = self.get_sosdisc_inputs('name')
        x_dict = self.get_sosdisc_inputs('x_dict')

        y_dict = {}
        for name_i, x_i in x_dict.items():
            y_dict[name_i] = a * x_i + b

        dict_values = {'indicator': a * b, 'y': a *
                                                x + b, 'y_dict': y_dict}

        self.store_sos_outputs_values(dict_values)
