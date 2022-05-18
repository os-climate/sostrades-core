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


class Disc1(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc1_doe_eval',
        'type': '',
        'source': '',
        'validated': '',
        'validated_by': '',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'x': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'a': {'type': 'float', 'unit': '-'},
        'b': {'type': 'float', 'unit': '-'}
    }
    DESC_OUT = {
        'indicator': {'type': 'float', 'unit': '-'},
        'y': {'type': 'float', 'unit': '-', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'z': {'type': 'float', 'unit': '-'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')
        a = self.get_sosdisc_inputs('a')
        b = self.get_sosdisc_inputs('b')
        dict_values = {'indicator': a * b, 'y': a * x + b, 'z': a - b}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
