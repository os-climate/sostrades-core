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


class Disc3(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc3_scenario',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-money-bill-alt fa-fw',
        'version': '',
    }
    _maturity = 'Fake'

    DESC_IN = {
        'z': {'type': 'float', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_disc3'},
        'constant': {'type': 'float', 'unit': '-'},
        'power': {'type': 'int', 'unit': '-'}
    }

    DESC_OUT = {
        'o': {'type': 'float', 'unit': '-', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_out_disc3'}
    }

    def run(self):
        z = self.get_sosdisc_inputs('z')
        constant = self.get_sosdisc_inputs('constant')
        power = self.get_sosdisc_inputs('power')
        dict_values = {'o': constant + z**power}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
