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


class DiscConvertGrad(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc_convertgrad',
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
        'gradient_outputs': {'type': 'dict', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_grad1'},
    }
    DESC_OUT = {
        'dzdx': {'type': 'float'},
        'dzdy': {'type': 'float'},
    }

    def run(self):
        input_dict = self.get_sosdisc_inputs('gradient_outputs')
#         dict_values = {'dzdx':input_dict[''] ,
#                        'dzdy':}
        dict_values = {}
        for key in input_dict:
            if key.endswith('.x'):
                dict_values['dzdx'] = input_dict[key]
            elif key.endswith('.y'):
                dict_values['dzdy'] = input_dict[key]
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

#     def _compute_jacobian(self, inputs=None, outputs=None):
#         a = self.get_sosdisc_inputs('a')
#         self.jac = {}
#         self.jac[outputs[0]] = {inputs[0]: array([[a]])}
