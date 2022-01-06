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


class Disc3(SoSDiscipline):
    _maturity = 'Fake'

    DESC_IN = {
        'z': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_disc3'},
        'constant': {'type': 'float'},
        'power': {'type': 'int'}
    }

    DESC_OUT = {
        'o': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_out_disc3'}
    }

    def run(self):
        z = self.get_sosdisc_inputs('z')
        constant = self.get_sosdisc_inputs('constant')
        power = self.get_sosdisc_inputs('power')
        dict_values = {'o': constant + z**power}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
