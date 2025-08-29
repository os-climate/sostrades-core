'''
Copyright 2024 Capgemini

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

from numpy import array, atleast_1d
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

"""
Implementation of a simple linear problem for optimization testing
"""


class LinearProblem(SoSWrapp):
    """Linear Problem for PuLP optimization testing"""

    _maturity = 'Fake'
    DESC_IN = {
        'x': {'type': 'int', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimLinear'},
        'y': {'type': 'int', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimLinear'},
    }

    DESC_OUT = {
        'objective': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimLinear'},
        'constraint_1': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimLinear'},
        'constraint_2': {'type': 'array', 'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_OptimLinear'},
    }

    def run(self):
        """Evaluates the linear problem at a given point"""
        x = self.get_sosdisc_inputs('x')
        y = self.get_sosdisc_inputs('y')

        # Objective: minimize 2*x + 3*y
        objective = 2.0 * x + 3.0 * y
        
        # Constraints:
        # x + y >= 4  -> -(x + y - 4) <= 0  -> 4 - x - y <= 0
        constraint_1 = 4.0 - x - y
        
        # 2*x + y >= 6 -> -(2*x + y - 6) <= 0 -> 6 - 2*x - y <= 0  
        constraint_2 = 6.0 - 2.0 * x - y

        # Store outputs as arrays (required by SOStrades)
        out = {
            'objective': atleast_1d(objective),
            'constraint_1': atleast_1d(constraint_1), 
            'constraint_2': atleast_1d(constraint_2)
        }
        self.store_sos_outputs_values(out)
