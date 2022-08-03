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
from sostrades_core.execution_engine.SoSWrapp import SoSWrapp
import numpy as np
from scipy.sparse import diags
# Discipline with dataframe


class Disc6(SoSWrapp):

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.disc6_wo_df',
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
        'x': {'type': 'array', 'visibility':  SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_protected'}
    }

    DESC_OUT = {
        'h': {'type': 'array', 'visibility':  SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_protected'}
    }

    def run(self):
        x = self.get_sosdisc_inputs('x')

        h = np.array([0.5 * (x[0] + 1. / (2 * x[0])),
                      0.5 * (x[1] + 1. / (2 * x[1]))])
        dict_values = {'h': h}
        self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        x = self.get_sosdisc_inputs('x')
        grad = [0.5 * (1.0 - 0.5 / x[0] ** 2), 0.5 * (1.0 - 0.5 / x[1] ** 2)]
        value = diags(grad) / 2
        self.set_partial_derivative('h', 'x', value)
