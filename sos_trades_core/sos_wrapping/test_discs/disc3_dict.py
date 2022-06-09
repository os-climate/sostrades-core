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

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc3_dict',
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
        'name_list': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_scatter_scenario'},
        'z': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'constant': {'type': 'float'},
        'power': {'type': 'int'}
    }

    DESC_OUT = {
        'o_dict': {'type': 'dict', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_scenario'}
    }

    def run(self):
        z = self.get_sosdisc_inputs('z')
        constant = self.get_sosdisc_inputs('constant')
        power = self.get_sosdisc_inputs('power')
        name_list = self.get_sosdisc_inputs('name_list')
        o_dict = {}
        i = 1
        for name in name_list:
            o_dict[name] = constant + i + z**power
            i += 1
        dict_values = {'o_dict': o_dict}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)
