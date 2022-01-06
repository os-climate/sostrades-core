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


class DiscHessian(SoSDiscipline):
    _maturity = 'Fake'
    DESC_IN = {
        'x': {'type': 'float'},
        'y': {'type': 'float'},
        'ax2': {'type': 'float'},
        'by2': {'type': 'float'},
        'cx': {'type': 'float'},
        'dy': {'type': 'float'},
        'exy': {'type': 'float'},
    }
    DESC_OUT = {
        'z': {'type': 'float'}
    }

    def run(self):
        input_dict = self.get_sosdisc_inputs()
        dict_values = {'z': input_dict['ax2'] * input_dict['x']**2 + input_dict['by2'] *
                       input_dict['y']**2 + input_dict['cx'] *
                       input_dict['x'] + input_dict['dy'] * input_dict['y']
                       + input_dict['exy'] * input_dict['x'] * input_dict['y']}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

#     def _compute_jacobian(self, inputs=None, outputs=None):
#         a = self.get_sosdisc_inputs('a')
#         self.jac = {}
#         self.jac[outputs[0]] = {inputs[0]: array([[a]])}
