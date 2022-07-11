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
import pandas as pd
import numpy as np


class Disc1(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc1_setup_sos_discipline',
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
        'AC_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [], 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac', 'structuring': True},
        'x': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'a': {'type': 'int'},
        'b': {'type': 'float'}
    }
    DESC_OUT = {
        'indicator': {'type': 'float'},
        'y': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'}
    }

    def setup_sos_disciplines(self):

        dynamic_inputs = {}
        dynamic_outputs = {}

        if 'AC_list' in self._data_in:
            AC_list = self.get_sosdisc_inputs('AC_list')

            for ac in AC_list:
                dynamic_inputs.update(
                    {f'{ac}.dyn_input_1': {'type': 'float', 'default': 0.}})
                dynamic_outputs.update({f'{ac}.dyn_output': {'type': 'float'}})
            default_df = pd.DataFrame(
                {'AC_name': AC_list, 'value': np.ones(len(AC_list))})
            dynamic_inputs.update(
                {'dyn_input_2': {'type': 'dataframe', 'default': default_df, 'structuring': True}})

            if 'dyn_input_2' in self._data_in and self.get_sosdisc_inputs('dyn_input_2')['AC_name'].to_list() != AC_list:
                self.update_default_value(
                    'dyn_input_2', self.IO_TYPE_IN, default_df)

        self.add_inputs(dynamic_inputs)
        self.add_outputs(dynamic_outputs)

    def run(self):
        input_dict = self.get_sosdisc_inputs()
        x = input_dict['x']
        a = input_dict['a']
        b = input_dict['b']
        dict_values = {'indicator': a * b, 'y': a * x + b}

        for ac in input_dict['AC_list']:
            dyn_input_ac = input_dict[f'{ac}.dyn_input_1']

            dyn_output = dyn_input_ac**2
            dict_values[f'{ac}.dyn_output'] = dyn_output
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
