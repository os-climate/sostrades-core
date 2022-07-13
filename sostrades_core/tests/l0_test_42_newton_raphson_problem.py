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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding: iso-8859-1 -*-
import unittest
from numpy import zeros, array
from os import remove
from sostrades_core.tools.grad_solvers.solvers.newton_raphson_problem import NewtonRaphsonProblem


class TestNewtonRaphsonProblem(unittest.TestCase):
    """
    NewtonRaphsonProblem test class
    """
    # Simple example to test the Newton Raphson method
    # Try to solve:
    #  R[0] = x**2+y**2
    #  R[1] = y**2+z**2
    #  R[2] = x**2+z**2
    #
    # solution expected [0.,0.,0.]

    def __comp_R(self, W):
        R = zeros(3)
        R[0] = W[0]**2 + W[1]**2
        R[1] = W[1]**2 + W[2]**2 + 10. * W[0] * W[1] * W[2]
        R[2] = W[0]**2 + W[2]**2
        return R

    def __comp_dRdW(self, W):
        dRdW = zeros((3, 3))

        dRdW[0, 0] = 2. * W[0]
        dRdW[0, 1] = 2. * W[1]
        dRdW[0, 2] = 0.

        dRdW[1, 0] = 10. * W[1] * W[2]
        dRdW[1, 1] = 2. * W[1] + 10. * W[0] * W[2]
        dRdW[1, 2] = 2. * W[2] + 10. * W[0] * W[1]

        dRdW[2, 0] = 2. * W[0]
        dRdW[2, 1] = 0.
        dRdW[2, 2] = 2. * W[2]
        return dRdW

    def __comp_wrong_dRdW(self, W):
        dRdW = zeros((3, 3))

        dRdW[0, 0] = 2. * W[0]
        dRdW[0, 1] = 2. * W[1]
        dRdW[0, 2] = 0.

        dRdW[1, 0] = 50. * W[1] * W[2]  # should be 10.*W[1]*W[2]
        dRdW[1, 1] = 2. * W[1] + 10. * W[0] * W[2]
        dRdW[1, 2] = 2. * W[2] + 10. * W[0] * W[1]

        dRdW[2, 0] = 2. * W[0]
        dRdW[2, 1] = 0.
        dRdW[2, 2] = 2. * W[2]
        return dRdW

    def test_01_NewtonRaphsonProblem_instantiation(self):
        """
        test class instantiation
        """
        W0 = array([1., 1., 1.])
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        assert(NRPb is not None)

    def test_02_NewtonRaphsonProblem_W0_list(self):
        """
        test set of starting point in python list format
        """
        W0 = [1., 1., 1.]
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        assert(NRPb is not None)

    def test_03_NewtonRaphsonProblem_set_relax_factor(self):
        """
        test relax_factor attribute overload
        """
        W0 = array([1., 1., 1.])
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        NRPb.set_relax_factor(0.80)
        assert(NRPb.get_relax_factor() == 0.80)

    def test_04_NewtonRaphsonProblem_set_stop_residual(self):
        """
        test stop_residual attribute overload
        """
        W0 = array([1., 1., 1.])
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        NRPb.set_stop_residual(1.e-9)
        assert(NRPb.get_stop_residual() == 1.e-9)

    def test_05_NewtonRaphsonProblem_set_max_iterations(self):
        """
        test max_iterations attribute overload
        """
        W0 = array([1., 1., 1.])
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        NRPb.set_max_iterations(200)
        assert(NRPb.get_max_iterations() == 200)

    def test_06_NewtonRaphsonProblem_defaults_parameters(self):
        """
        test defaults parameters sfor relax_factor, stop_residual and max_iterations
        """
        W0 = array([1., 1., 1.])
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        assert(NRPb.get_relax_factor() == 0.99)
        assert(NRPb.get_stop_residual() == 1.e-6)
        assert(NRPb.get_max_iterations() == 100)

    def test_07_NewtonRaphsonProblem_jacobian_validation(self):
        """
        test Hessian validation when Hessian is valid
        """
        W0 = array([1., 1., 1.])
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        ok = NRPb.valid_jacobian(W0, iprint=True)
        assert(ok)

        remove('gradient_file0.dat')
        remove('gradient_file1.dat')
        remove('gradient_file2.dat')

    def test_08_NewtonRaphsonProblem_jacobian_non_validation(self):
        """
        test Hessian validation when Hessian is not valid
        """
        W0 = array([1., 1., 1.])
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_wrong_dRdW)
        ok, df_fd, df = NRPb.valid_jacobian(W0, iprint=False)
        assert(not ok)

    def test_09_NewtonRaphsonProblem_solving(self):
        W0 = array([1., 1., 1.])
        sol = array([0., 0., 0.])
        print('')
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        NRPb.set_relax_factor(0.99)
        NRPb.set_stop_residual(1.e-15)
        NRPb.set_max_iterations(100)
        W = NRPb.solve()
        self.assertAlmostEqual(W[0], sol[0])
        self.assertAlmostEqual(W[1], sol[1])
        self.assertAlmostEqual(W[2], sol[2])

    def test_10_NewtonRaphsonProblem_solving_limit_max_iterations(self):
        W0 = array([1., 1., 1.])
        sol = array([0., 0., 0.])
        print('')
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        NRPb.set_relax_factor(0.99)
        NRPb.set_stop_residual(1.e-15)
        NRPb.set_max_iterations(10)
        W = NRPb.solve()
        res_hist = NRPb.get_residual_hist()
        assert(len(res_hist) == 10)

    def test_11_NewtonRaphsonProblem_solving_limit_residual(self):
        W0 = array([1., 1., 1.])
        sol = array([0., 0., 0.])
        print('')
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        NRPb.set_relax_factor(0.99)
        NRPb.set_stop_residual(1.e-6)
        NRPb.set_max_iterations(100)
        W = NRPb.solve()
        res = NRPb.get_residual()
        assert(1.e-7 < res < 1.e-6)

    def test_12_NewtonRaphsonProblem_solving_parallel(self):
        W0 = array([1., 1., 1.])
        sol = array([0., 0., 0.])
        print('')
        NRPb = NewtonRaphsonProblem(W0, self.__comp_R, self.__comp_dRdW)
        NRPb.set_relax_factor(0.99)
        NRPb.set_stop_residual(1.e-15)
        NRPb.set_max_iterations(100)
        NRPb.set_method = 'inhouse'
        NRPb.multi_proc = True
        W = NRPb.solve()
        self.assertAlmostEqual(W[0], sol[0])
        self.assertAlmostEqual(W[1], sol[1])
        self.assertAlmostEqual(W[2], sol[2])


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNewtonRaphsonProblem)
    unittest.TextTestRunner(verbosity=2).run(suite)
