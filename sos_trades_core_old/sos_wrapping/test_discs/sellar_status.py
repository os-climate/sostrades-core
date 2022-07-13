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
from copy import copy
from cmath import exp, sqrt
from numpy import array, atleast_2d, NaN, complex128, ones, zeros
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class SellarProblem(SoSDiscipline):
    """ Sellar Optimization Problem functions
    """
    _maturity = 'Fake'
    DESC_IN = {'x': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_1': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_2': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'local_dv': {'type': 'float','range': [0.0, 20.0]}}

    DESC_OUT = {'c_1': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                'c_2': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
                'obj': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ computes
        """
        x, y_1, y_2, z = self.get_sosdisc_inputs(['x', 'y_1', 'y_2', 'z'])
        local_dv = self.get_sosdisc_inputs('local_dv')

        obj = self.obj(x, z, y_1, y_2)
        c_1 = self.c_1(y_1)
        c_2 = self.c_2(y_2)
        obj += local_dv
        out = {'obj': array([obj]), 'c_1': array([c_1]), 'c_2': array([c_2])}
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
        out = x ** 2 + z[1] + y_1 + exp(-y_2)

        return out

    @staticmethod
    def c_1(y_1):
        """First constraint on system level

        :param y_1: coupling variable from discipline 1
        :type y_1: numpy.array
        :returns: Value of the constraint 1
        :rtype: float
        """
        return 3.16 - y_1

    @staticmethod
    def c_2(y_2):
        """Second constraint on system level

        :param y_2: coupling variable from discipline 2
        :type y_2: numpy.array
        :returns: Value of the constraint 2
        :rtype: float
        """
        return y_2 - 24.

    def compute_sos_jacobian(self):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

        x, y_2 = self.get_sosdisc_inputs(['x', 'y_2'])

        self.set_partial_derivative('c_1', 'y_1', atleast_2d(array([-1.0])))

        self.set_partial_derivative('c_2', 'y_2', atleast_2d(array(
            [1.0])))

        self.set_partial_derivative('obj', 'x', atleast_2d(array(
            [2.0 * x])))

        self.set_partial_derivative('obj', 'z', atleast_2d(array(
            [0.0, 1.0])))
        self.set_partial_derivative('obj', 'y_1', atleast_2d(array(
            [1.0])))
        self.set_partial_derivative('obj', 'y_2', atleast_2d(array(
            [-exp(-y_2)])))

        self.set_partial_derivative('obj', 'local_dv', atleast_2d(array(
            [1.0])))


class Sellar1(SoSDiscipline):
    """ Discipline 1
    """
    _maturity = 'Fake'
    DESC_IN = {'x': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'y_2': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {'y_1': {'type': 'float',
                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ Discipline 1 execution
        """
        x, y_2, z = self.get_sosdisc_inputs(['x', 'y_2', 'z'])
        y_1 = self.compute_y_1(x, y_2, z)
        y1_out = {'y_1': y_1}
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_1(x, y_2, z):
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
        out = z[0] ** 2 + x + z[1] - 0.2 * y_2
        return out

    def compute_sos_jacobian(self):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

        z = self.get_sosdisc_inputs('z')

        self.set_partial_derivative('y_1', 'x', atleast_2d(array([1.0])))

        self.set_partial_derivative('y_1', 'z', atleast_2d(array(
            [2.0 * z[0], 1.0])))

        self.set_partial_derivative('y_1', 'y_2', atleast_2d(array([-0.2])))


class Sellar2(SoSDiscipline):
    """ Discipline 2
    """

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.sellar',
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
    DESC_IN = {'y_1': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'debug_mode_sellar': {'type': 'bool', 'default':False, 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {'y_2': {'type': 'float',
                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ solves Discipline1
        """
        y_1, z,debug_mode = self.get_sosdisc_inputs(['y_1', 'z','debug_mode_sellar'])
        y_2 = self.compute_y_2(y_1, z)
        y1_out = {'y_2': y_2}
        self.store_sos_outputs_values(y1_out)

        if debug_mode:
            # if debug mode activated raise an error
            raise Exception("debug mode activated to trigger except")



    @staticmethod
    def compute_y_2(y_1, z):
        """Solve the second coupling equation in functional form.

        :param z: vector of shared design variables
        :type z: numpy.array
        :param y_1: coupling variable of discipline 1
        :type y_1: numpy.array
        :returns: coupling variable y_2
        :rtype: float
        """
        out = z[0] + z[1] + sqrt(y_1)
        return out

    def compute_sos_jacobian(self):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

        y_1, debug_mode = self.get_sosdisc_inputs(['y_1', 'debug_mode_sellar'])

        self.set_partial_derivative('y_2', 'y_1', atleast_2d(
            array([1.0 / (2.0 * sqrt(y_1))])))

        self.set_partial_derivative('y_2', 'z', atleast_2d(
            array([1.0, 1.0])))

        if debug_mode:
            # if debug mode activated raise an error
            raise Exception("debug mode activated to trigger except")


class Sellar3(SoSDiscipline):
    """ Discipline 2 but with NaN in calculation on purpose for test
    """

    # ontology information
    _ontology_data = {
        'label': 'sostrades_core.sos_wrapping.test_discs.sellar',
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
    DESC_IN = {'y_1': {'type': 'float', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array', 'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'},
               'error_string': {'type': 'string', 'default': '', 'possible_values': ["", "nan", "input_change",
                            "linearize_data_change", "min_max_grad", "min_max_couplings", "all"]}}

    DESC_OUT = {'y_2': {'type': 'float',
                        'visibility': SoSDiscipline.SHARED_VISIBILITY, 'namespace': 'ns_OptimSellar'}}

    def run(self):
        """ computes Discipline3
        """
        y_1, z = self.get_sosdisc_inputs(['y_1', 'z'])
        error_string = self.get_sosdisc_inputs('error_string')

        y_2 = self.compute_y_2(y_1, z)
        y1_out = {'y_2': y_2}
        if error_string == 'nan':
            y1_out['y_2'] = NaN
        elif error_string == 'input_change':
            y_1 = self.local_data[self.get_var_full_name('y_1', self._data_in)]
            y_1 += 0.5
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_2(y_1, z):
        """Solve the second coupling equation in functional form.

        :param z: vector of shared design variables
        :type z: numpy.array
        :param y_1: coupling variable of discipline 1
        :type y_1: numpy.array
        :returns: coupling variable y_2
        :rtype: float
        """
        out = z[0] + z[1] + sqrt(y_1)
        return out

    def compute_sos_jacobian(self):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """

        y_1 = self.get_sosdisc_inputs(['y_1'])
        error_string = self.get_sosdisc_inputs('error_string')

        if error_string == 'linearize_data_change':
            y_1 = self.local_data[self.get_var_full_name('y_1', self._data_in)]
            y_1 += 0.5

        self.set_partial_derivative('y_2', 'y_1', atleast_2d(
            array([1.0 / (2.0 * sqrt(y_1))])))

        self.set_partial_derivative('y_2', 'z', atleast_2d(
            array([1.0, 1.0])))

        if error_string == 'min_max_grad':
            self.set_partial_derivative('y_2', 'y_1', atleast_2d(
                array([1e10])))

    def _check_min_max_gradients(self, jac):
        '''Override the _check_min_max_gradients method from <gemseo.core.discipline> with a raise for test purposes
        THIS METHOD MUST BE UPDATED IF THE ORIGINAL METHOD CHANGES
        '''
        from numpy import min as np_min
        from numpy import max as np_max

        for out in jac:
            for inp in self.jac[out]:
                grad = self.jac[out][inp]
                # avoid cases when gradient is not required
                if grad.size > 0:
                    d_name = self.name
                    #                     cond_number = np.linalg.cond(grad)
                    #                     if cond_number > 1e10 and not np.isinf(cond_number):
                    #                         self.logger.info(
                    # f'The Condition number of the jacobian dr {out} / dr {inp} is
                    # {cond_number}')
                    mini = np_min(grad.toarray())
                    if mini < -1e4:
                        self.ee.logger.info(
                            "in discipline <%s> : dr<%s> / dr<%s>: minimum gradient value is <%s>" % (
                                d_name, out, inp, mini))

                    maxi = np_max(grad.toarray())
                    if maxi > 1e4:
                        self.ee.logger.info(
                            "in discipline <%s> : dr<%s> / dr<%s>: maximum gradient value is <%s>" % (
                                d_name, out, inp, maxi))
                        raise ValueError("in discipline <%s> : dr<%s> / dr<%s>: maximum gradient value is <%s>" % (
                                d_name, out, inp, maxi))

    def display_min_max_couplings(self):
        ''' Override the display_min_max_couplings method from <sostrades_core.execution_engine.sos_discpline> with a raise for test purposes
            THIS METHOD MUST BE UPDATED IF THE ORIGINAL METHOD CHANGES
        '''
        coupling_dict = {}
        for key, value in self.local_data.items():
            is_coupling = self.dm.get_data(key, 'coupling')
            if is_coupling:
                coupling_dict[key] = value
        min_coupling = min(coupling_dict, key=coupling_dict.get)
        max_coupling = max(coupling_dict, key=coupling_dict.get)
        self.ee.logger.info(
            "in discipline <%s> : <%s> has the minimum coupling value <%s>" % (
                self.sos_name, min_coupling, coupling_dict[min_coupling]))
        self.ee.logger.info(
            "in discipline <%s> : <%s> has the maximum coupling value <%s>" % (
                self.sos_name, max_coupling, coupling_dict[max_coupling]))
        raise ValueError("in discipline <%s> : <%s> has the minimum coupling value <%s>" % (
                self.sos_name, min_coupling, coupling_dict[min_coupling]))


if __name__ == '__main__':
    disc_id = 'coupling_disc'
    namespace = "test"
    ee = ExecutionEngine("test")
