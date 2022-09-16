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
import numpy as np


class Polynom(SoSDiscipline):

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.disc1',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-plane fa-fw',
        'version': '',
    }
    _maturity = 'Fake'
    DESC_IN = {
        'newton_unknowns': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_z'},
        'float_unknown': {'type': 'float', 'default': 1.0, 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_float'},
    }
    DESC_OUT = {
        'z': {'type': 'array', 'unit': '-', 'visibility': 'Shared', 'namespace': 'ns_z'},
    }

    def run(self):
        newton_unknowns = self.get_sosdisc_inputs('newton_unknowns')
        x = newton_unknowns[0]
        y = newton_unknowns[1]

        dict_values = {'z': np.array([x - y + 1, y - x**2 - 1])}
        # put new field value in data_out
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        newton_unknowns = self.get_sosdisc_inputs('newton_unknowns')

        grad = np.array([[1, -1],
                         [-2 * newton_unknowns[0], 1]])
        self.set_partial_derivative('z', 'newton_unknowns', grad)
