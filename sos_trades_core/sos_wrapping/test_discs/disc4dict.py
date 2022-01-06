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


class Disc4(SoSDiscipline):
    _maturity = 'Fake'

    DESC_IN = {
        'h': {'type': 'array', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'mydict': {'type': 'dict'},
    }

    DESC_OUT = {
        'z': {'type': 'array', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'},
        'dict_out': {'type': 'dict', 'visibility':  SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_test'}
    }

    def run(self):
        mydict = self.get_sosdisc_inputs('mydict')
        h = self.get_sosdisc_inputs('h')
        dict_out = {'key1': ((h[0] + h[1]) / 2),
                    'key2': ((h[0] + h[1]) / 2)}
        z = array([h[0], 2 * h[1]])

        dict_values = {'z': z,
                       'dict_out': dict_out}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
