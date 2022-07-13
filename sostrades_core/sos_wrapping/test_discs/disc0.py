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
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline


class Disc0(ProxyDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc0',
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
        'r': {'type': 'float'},
        'mod': {'type': 'int', 'default': 1}
    }
    DESC_OUT = {
        'x': {'type': 'float', 'visibility': 'Shared', 'namespace': 'ns_disc1'},
        'a': {'type': 'int', 'visibility': 'Shared', 'namespace': 'ns_disc1'}
    }

    def run(self):

        r = self.get_sosdisc_inputs('r')
        mod = self.get_sosdisc_inputs('mod')
        a, x = divmod(r, mod)

        dict_values = {'a': int(a), 'x': x}

        self.store_sos_outputs_values(dict_values)
