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

from cmath import exp, sqrt
from numpy import array, atleast_2d, complex128, ones, zeros
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class SellarProblem(SoSDiscipline):
    """ Sellar Optimization Problem functions
    """
    _maturity = 'Fake'
    DESC_IN = {'x': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_01': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_10': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_02': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_20': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'local_dv': {'type': 'float'}}

    DESC_OUT = {'c_1': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                'c_2': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                'c_3': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                'c_4': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                'obj': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ computes
        """
        x, y_01, y_10, y_02, y_20, z = self.get_sosdisc_inputs(
            ['x', 'y_01', 'y_10', 'y_02', 'y_20', 'z'])
        local_dv = self.get_sosdisc_inputs('local_dv')

        obj = self.obj(x, z, y_10, y_20)
        c_1 = self.c_1(y_10)
        c_2 = self.c_2(y_20)
        c_3 = self.c_eq(y_02, y_10)
        c_4 = self.c_eq(y_20, y_01)
        obj += local_dv
        out = {'obj': obj, 'c_1': c_1, 'c_2': c_2, 'c_3': c_3, 'c_4': c_4}
        self.store_sos_outputs_values(out)

    @staticmethod
    def obj(x, z, y_1, y_2):
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
        out = x[0] ** 2 + z[1] + y_1[0] + exp(-y_2[0])
        return array([out])

    @staticmethod
    def c_1(y_1):
        """First constraint on system level

        :param y_1: coupling variable from discipline 1
        :type y_1: numpy.array
        :returns: Value of the constraint 1
        :rtype: float
        """
        return array([3.16 - y_1[0]])

    @staticmethod
    def c_2(y_2):
        """Second constraint on system level

        :param y_2: coupling variable from discipline 2
        :type y_2: numpy.array
        :returns: Value of the constraint 2
        :rtype: float
        """
        return array([y_2[0] - 24.])

    @staticmethod
    def c_eq(y_02, y_10):
        """Second constraint on system level

        :param y_2: coupling variable from discipline 2
        :type y_2: numpy.array
        :returns: Value of the constraint 2
        :rtype: float
        """
        return array([(y_02[0] - y_10[0])**2])


class Sellar1(SoSDiscipline):
    """ Discipline 1
    """
    _maturity = 'Fake'
    DESC_IN = {'x': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_01': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {'y_10': {'type': 'array',
                         'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ Discipline 1 execution
        """
        x, y_01, z = self.get_sosdisc_inputs(['x', 'y_01', 'z'])
        y_10 = self.compute_y_1(x, y_01, z)
        y1_out = {'y_10': y_10}
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_1(x, y_01, z):
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
        out = z[0] ** 2 + x[0] + z[1] - 0.2 * y_01[0]
        return array([out])


class Sellar2(SoSDiscipline):
    """ Discipline 2
    """

    # ontology information
    _ontology_data = {
        'label': 'sos_trades_core.sos_wrapping.test_discs.sellar_idf',
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
    DESC_IN = {'y_02': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {'y_20': {'type': 'array',
                         'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ solves Discipline1
        """
        y_02, z = self.get_sosdisc_inputs(['y_02', 'z'])
        y_20 = self.compute_y_2(y_02, z)
        y1_out = {'y_20': y_20}
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_2(y_02, z):
        """Solve the second coupling equation in functional form.

        :param z: vector of shared design variables
        :type z: numpy.array
        :param y_1: coupling variable of discipline 1
        :type y_1: numpy.array
        :returns: coupling variable y_2
        :rtype: float
        """
        out = z[0] + z[1] + sqrt(y_02[0])
        return array([out])


if __name__ == '__main__':
    disc_id = 'coupling_disc'
    namespace = "test"
    ee = ExecutionEngine("test")
