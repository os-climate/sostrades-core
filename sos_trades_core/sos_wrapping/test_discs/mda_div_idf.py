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

from numpy import array
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from copy import deepcopy


class SellarProblem(SoSDiscipline):
    """ Sellar Optimization Problem functions
    """
    _maturity = 'Fake'
    DESC_IN = {
        'y_1': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
        'y_2': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
        'y_1_cp': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
        'y_2_cp': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {
        'c_3': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
        'c_4': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
        'obj': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ computes
        """
        y_1, y_2, y_1_cp, y_2_cp = self.get_sosdisc_inputs(
            ['y_1', 'y_2', 'y_1_cp', 'y_2_cp'])
        obj = self.obj(y_1, y_2)
        c_3 = self.c_eq(y_1, y_1_cp)
        c_4 = self.c_eq(y_2, y_2_cp)

        out = {'obj': obj,  'c_3': c_3, 'c_4': c_4}
        self.store_sos_outputs_values(out)

    @staticmethod
    def obj(x, y):
        """Objective function

        :param x: local design variables
        :type x: numpy.array
        :param z: shared design variables
        :type z: numpy.array
        :param y_1: coupling variable from discipline 1
        :type y_1: numpy.array
        :param y_2: coupling variable from discipline 2
        :type y_2: numpy.array
        :returns: Objective value
        :rtype: float
        """
        out = -x[0] + 2 * y[0]
        return array([out])

    @staticmethod
    def c_eq(y_02, y_10):
        """Second constraint on system level

        :param y_2: coupling variable from discipline 2
        :type y_2: numpy.array
        :returns: Value of the constraint 2
        :rtype: float
        """
        return array([(y_02[0] - y_10[0])])


class Sellar1(SoSDiscipline):
    """ Discipline 1
    """
    _maturity = 'Fake'
    DESC_IN = {
        'y_1': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {'y_2': {'type': 'array',
                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ Discipline 1 execution
        """
        y_01 = self.get_sosdisc_inputs('y_1')
        y_10 = self.compute_y_1(y_01)
        y1_out = {'y_2': y_10}
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_1(x):
        """Solve the first coupling equation in functional form.

        :param x: vector of design variables local to discipline 1
        :type x: numpy.array
        :param z: vector of shared design variables
        :type z: numpy.array
        :param y_2: coupling variable of discipline 2
        :type y_2: numpy.array
        :returns: coupling variable y_1 of discipline 1
        :rtype: float
        """
        out = x[0] ** 2 - 5
        return array([out])


class Sellar2(SoSDiscipline):
    """ Discipline 2
    """
    _maturity = 'Fake'
    DESC_IN = {'y_2': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               }

    DESC_OUT = {'y_1': {'type': 'array',
                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ solves Discipline1
        """
        y_2 = self.get_sosdisc_inputs('y_2')
        y_20 = self.compute_y_2(y_2)
        y1_out = {'y_1': y_20}
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_2(y_02):
        """Solve the second coupling equation in functional form.

        :param z: vector of shared design variables
        :type z: numpy.array
        :param y_1: coupling variable of discipline 1
        :type y_1: numpy.array
        :returns: coupling variable y_2
        :rtype: float
        """
        out = y_02[0]
        return array([out])


class SellarCopy(SoSDiscipline):
    """ Discipline 2
    """

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.mda_div_idf',
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
    DESC_IN = {'y_1': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_2': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               }

    DESC_OUT = {'y_1_cp': {'type': 'array',
                           'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                'y_2_cp': {'type': 'array',
                           'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                }

    def run(self):
        """ solves Discipline1
        """
        y_1, y_2 = self.get_sosdisc_inputs(['y_1', 'y_2'])
        y_1_cp = deepcopy(y_1)
        y_2_cp = deepcopy(y_2)
        y1_out = {'y_1_cp': y_1_cp, 'y_2_cp': y_2_cp}
        self.store_sos_outputs_values(y1_out)


if __name__ == '__main__':
    disc_id = 'coupling_disc'
    namespace = "test"
    ee = ExecutionEngine("test")
