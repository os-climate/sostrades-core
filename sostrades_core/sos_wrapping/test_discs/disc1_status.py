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
from sostrades_core.execution_engine.discipline_proxy import ProxyDiscipline
import time


class Disc1status(ProxyDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc1_status',
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
        'x': {'type': 'float', 'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'},
        'a': {'type': 'float'},
        'b': {'type': 'float'}
    }
    DESC_OUT = {
        'indicator': {'type': 'float'},
        'y': {'type': 'float', 'visibility': ProxyDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_ac'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')
        a = self.get_sosdisc_inputs('a')
        b = self.get_sosdisc_inputs('b')
#         info = self.get_sosdisc_inputs('info')
        dict_values = {'indicator': a * b, 'y': a * x + b}
        time.sleep(0.2)
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

#     def _compute_jacobian(self, inputs=None, outputs=None):
#         a = self.get_sosdisc_inputs('a')
#         self.jac = {}
#         self.jac[outputs[0]] = {inputs[0]: array([[a]])}
