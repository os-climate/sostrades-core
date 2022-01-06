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


class Disc2(SoSDiscipline):
    _maturity = 'Fake'
    DESC_IN = {
        'y_dict': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_barrier'},
        'x_dict': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_barrier'},
        'constant': {'type': 'float'},
        'power': {'type': 'int'},
    }
    DESC_OUT = {
        'z_dict': {'type': 'dict', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'}
    }

    def run(self):
        y_dict = self.get_sosdisc_inputs('y_dict')
        x_dict = self.get_sosdisc_inputs('x_dict')
        constant = self.get_sosdisc_inputs('constant')
        power = self.get_sosdisc_inputs('power')

        z_dict = {}
        for key, val in y_dict.items():
            z_dict[key] = constant + val**power + x_dict[key]

        dict_values = {'z_dict': z_dict}

        self.store_sos_outputs_values(dict_values)
