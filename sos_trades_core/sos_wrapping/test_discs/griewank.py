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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8 -*-
'''
Implementation of Sellar Disciplines (Sellar, 1996)
Adapted from GEMSEO examples
'''

import math
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class Griewank(SoSDiscipline):
    """ Griewank Optimization Problem functions
    """

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.griewank',
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
    DESC_IN = {'chromosome': {'type': 'array',
                              'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimGriewank'}}

    DESC_OUT = {
        'obj': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimGriewank'}}

    def run(self):
        """ computes Griewank function
        """
        chromosome = self.get_sosdisc_inputs('chromosome')

        part1 = 0
        for i in range(len(chromosome)):
            part1 += chromosome[i]**2
            part2 = 1
        for i in range(len(chromosome)):
            part2 *= math.cos(float(chromosome[i]) / math.sqrt(i + 1))
        obj = 1 + (float(part1) / 4000.0) - float(part2)
        self.store_sos_outputs_values({'obj': obj})


if __name__ == '__main__':
    disc_id = 'coupling_disc'
    namespace = "test"
    ee = ExecutionEngine("test")
