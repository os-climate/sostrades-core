'''
Copyright 2022 Airbus SAS
Modifications on 2024/05/16 Copyright 2024 Capgemini

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
from cmath import exp as exp_cp
from cmath import sqrt as sqrt_cp

from numpy import NaN, array, atleast_2d, floating
from numpy import exp as exp_np
from numpy import sqrt as sqrt_np

from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

'''
Implementation of Sellar Disciplines (Sellar, 1996)
Adapted from GEMSEO examples
'''


def sqrt_complex(y):
    """
    Returns the square root of a given number. If the number is non-negative and of type 'floating',
    it computes the square root using a regular NumPy function. Otherwise, it uses a custom function for complex numbers.

    Args:
        y (float or complex): The number for which to compute the square root.

    Returns:
        float or complex: The square root of the input number.

    """
    if isinstance(y, floating) and y >= 0:
        return sqrt_np(y)
    else:
        return sqrt_cp(y)


def exp_complex(y):
    """
    Returns the exponential of a given number. If the number is non-negative and of type 'floating',
    it computes the exponential using a regular NumPy function. Otherwise, it uses a custom function for complex numbers.

    Args:
        y (float or complex): The number for which to compute the exponential.

    Returns:
        float or complex: The exponential of the input number.

    """
    if isinstance(y, floating) and y >= 0:
        return exp_np(y)
    else:
        return exp_cp(y)


class SellarProblem(SoSWrapp):
    """Sellar Optimization Problem functions"""

    _maturity = 'Fake'
    DESC_IN = {'x': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'y_1': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'y_2': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'local_dv': {'type': 'float'}}

    DESC_OUT = {'c_1': {'type': 'array',  'namespace': 'ns_OptimSellar'},
                'c_2': {'type': 'array',  'namespace': 'ns_OptimSellar'},
                'obj': {'type': 'array',  'namespace': 'ns_OptimSellar'}}

    def run(self):
        """Computes"""
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
        """
        Objective function

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
        out = x[0] ** 2 + z[1] + y_1[0] + exp_complex(-y_2[0])

        return out

    @staticmethod
    def c_1(y_1):
        """
        First constraint on system level

        :param y_1: coupling variable from discipline 1
        :type y_1: numpy.array
        :returns: Value of the constraint 1
        :rtype: float
        """
        return 3.16 - y_1[0]

    @staticmethod
    def c_2(y_2):
        """
        Second constraint on system level

        :param y_2: coupling variable from discipline 2
        :type y_2: numpy.array
        :returns: Value of the constraint 2
        :rtype: float
        """
        return y_2[0] - 24.

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
            [2.0 * x[0]])))

        self.set_partial_derivative('obj', 'z', atleast_2d(array(
            [0.0, 1.0])))
        self.set_partial_derivative('obj', 'y_1', atleast_2d(array(
            [1.0])))
        self.set_partial_derivative('obj', 'y_2', atleast_2d(array(
            [-exp_complex(-y_2[0])])))

        self.set_partial_derivative('obj', 'local_dv', atleast_2d(array(
            [1.0])))


class Sellar1(SoSWrapp):
    """Discipline 1"""

    _maturity = 'Fake'
    DESC_IN = {'x': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'y_2': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array',  'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {'y_1': {'type': 'array',
                         'namespace': 'ns_OptimSellar'}}

    def run(self):
        """Discipline 1 execution"""
        x, y_2, z = self.get_sosdisc_inputs(['x', 'y_2', 'z'])
        y_1 = self.compute_y_1(x, y_2, z)
        y1_out = {'y_1': array([y_1])}
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_1(x, y_2, z):
        """
        Solve the first coupling equation in functional form.

        :param x: vector of design variables local to discipline 1
        :type x: numpy.array
        :param z: vector of shared design variables
        :type z: numpy.array
        :param y_2: coupling variable of discipline 2
        :type y_2: numpy.array
        :returns: coupling variable y_1 of discipline 1
        :rtype: float
        """
        out = z[0] ** 2 + x[0] + z[1] - 0.2 * y_2[0]
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


class Sellar2(SoSWrapp):
    """Discipline 2"""

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
    DESC_IN = {'y_1': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'debug_mode_sellar': {'type': 'bool', 'default': False,
                                     'namespace': 'ns_OptimSellar'}}

    DESC_OUT = {'y_2': {'type': 'array',
                         'namespace': 'ns_OptimSellar'}}

    def run(self):
        """Solves Discipline1"""
        y_1, z = self.get_sosdisc_inputs(['y_1', 'z'])
        y_2 = self.compute_y_2(y_1, z)
        y1_out = {'y_2': array([y_2])}
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_2(y_1, z):
        """
        Solve the second coupling equation in functional form.

        :param z: vector of shared design variables
        :type z: numpy.array
        :param y_1: coupling variable of discipline 1
        :type y_1: numpy.array
        :returns: coupling variable y_2
        :rtype: float
        """
        out = z[0] + z[1] + sqrt_complex(y_1[0])
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
            array([1.0 / (2.0 * sqrt_complex(y_1[0]))])))

        self.set_partial_derivative('y_2', 'z', atleast_2d(
            array([1.0, 1.0])))

        if debug_mode:
            # if debug mode activated raise an error
            raise Exception("debug mode activated to trigger except")


class Sellar3(SoSWrapp):
    """Discipline 2 but with NaN in calculation on purpose for test"""

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
    DESC_IN = {'y_1': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'z': {'type': 'array',  'namespace': 'ns_OptimSellar'},
               'error_string': {'type': 'string', 'default': '', 'possible_values': ["", "nan", "input_change",
                                                                                     "linearize_data_change",
                                                                                     "min_max_grad",
                                                                                     "min_max_couplings", "all"]}}

    DESC_OUT = {'y_2': {'type': 'array',
                         'namespace': 'ns_OptimSellar'}}

    def run(self):
        """Computes Discipline3"""
        y_1, z = self.get_sosdisc_inputs(['y_1', 'z'])
        error_string = self.get_sosdisc_inputs('error_string')

        y_2 = self.compute_y_2(y_1, z)
        y1_out = {'y_2': array([y_2])}
        if error_string == 'nan':
            y1_out['y_2'] = array([NaN])
            raise Exception('error test')
        elif error_string == 'input_change':
            y_1 = self.get_sosdisc_inputs('y_1')
            y_1[0] += 0.5
        self.store_sos_outputs_values(y1_out)

    @staticmethod
    def compute_y_2(y_1, z):
        """
        Solve the second coupling equation in functional form.

        :param z: vector of shared design variables
        :type z: numpy.array
        :param y_1: coupling variable of discipline 1
        :type y_1: numpy.array
        :returns: coupling variable y_2
        :rtype: float
        """
        out = z[0] + z[1] + sqrt_complex(y_1[0])
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
        y_1 = self.get_sosdisc_inputs('y_1')
        error_string = self.get_sosdisc_inputs('error_string')

        if error_string == 'linearize_data_change':
            y_1_fullname = [full_name for full_name in self.io.data.keys() if
                            'y_1' == full_name.split('.')[-1]][0]
            y_1 = self.io.data[y_1_fullname]
            y_1[0] += 0.5

        self.set_partial_derivative('y_2', 'y_1', atleast_2d(
            array([1.0 / (2.0 * sqrt_complex(y_1[0]))])))

        self.set_partial_derivative('y_2', 'z', atleast_2d(
            array([1.0, 1.0])))

        if error_string == 'min_max_grad':
            self.set_partial_derivative('y_2', 'y_1', atleast_2d(
                array([1e10])))

    def _check_min_max_gradients(self, jac):
        '''
        Override the _check_min_max_gradients method from <gemseo.core.discipline> with a raise for test purposes
        THIS METHOD MUST BE UPDATED IF THE ORIGINAL METHOD CHANGES
        '''
        from numpy import max as np_max
        from numpy import min as np_min

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


if __name__ == '__main__':
    disc_id = 'coupling_disc'
    namespace = "test"
    ee = ExecutionEngine("test")
