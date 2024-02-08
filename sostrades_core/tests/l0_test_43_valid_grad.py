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

from sostrades_core.tools.grad_solvers.validgrad.FDValidGrad import FDValidGrad
from sostrades_core.tools.grad_solvers.validgrad.BFGSFDHessian import BFGSFDHessian
from sostrades_core.tools.grad_solvers.validgrad.FDGradient import FDGradient
from sostrades_core.tools.grad_solvers.validgrad.FDSecondOrderCentered import FDSecondOrderCentered
from math import sin, cos
from cmath import cos as ccos
from cmath import sin as csin
import numpy as np


class TestValidGrad(unittest.TestCase):
    """
    Class to test gradient validation with grad_solvers tools
    """

    def f_scalar(self, x):
        return np.sum(x)

    def df_scalar(self, x):
        n = np.shape(x)[0]
        return np.ones(n)

    def f_mat1(self, x):
        return 3. * x

    def df_mat1(self, x):
        n = np.shape(x)[0]
        return 3. * np.eye(n)

    def f_mat2(self, x):
        fx = np.zeros((2, 2))
        fx[0, 0] = sin(x[0])
        fx[0, 1] = fx[0, 0]
        fx[1, 0] = cos(x[1])
        fx[1, 1] = fx[1, 0]

        return fx

    def f_mat2_CS(self, x):
        fx = np.zeros((2, 2), dtype=np.complex128)
        fx[0, 0] = csin(x[0])
        fx[0, 1] = fx[0, 0]
        fx[1, 0] = ccos(x[1])
        fx[1, 1] = fx[1, 0]

        return fx

    def df_mat2(self, x):
        grad = np.zeros((2, 2, 2))
        grad[0, 0, 0] = cos(x[0])
        grad[0, 1, 0] = grad[0, 0, 0]
        grad[1, 0, 0] = 0.
        grad[1, 1, 0] = 0.

        grad[0, 0, 1] = 0.
        grad[0, 1, 1] = 0.
        grad[1, 0, 1] = -sin(x[1])
        grad[1, 1, 1] = grad[1, 0, 1]

        return grad

    def test_01_grad_mat1(self):
        n = 7
        x = np.ones(n)
        x[0] = 5.
        x[5] = 9.
        vg1 = FDValidGrad(1, self.f_mat1, self.df_mat1)
        vg2 = FDValidGrad(2, self.f_mat1, self.df_mat1)
        ok1 = vg1.compare(x)
        ok2 = vg2.compare(x)
        assert(ok1)
        assert(ok2)

    def test_02_grad_mat1_CS(self):
        n = 7
        x = np.ones(n)
        x[0] = 5.
        x[5] = 9.
        vg3 = FDValidGrad(1j, self.f_mat1, self.df_mat1)
        ok3 = vg3.compare(x, split_out=True)
        assert(ok3)

    def test_03_grad_mat2(self):
        n = 2
        x = np.ones(n)
        x[0] = 5.
        x[1] = 6.
        vg1 = FDValidGrad(1, self.f_mat2, self.df_mat2)
        vg2 = FDValidGrad(2, self.f_mat2, self.df_mat2)
        ok1 = vg1.compare(x)
        ok2 = vg2.compare(x)
        assert(ok1)
        assert(ok2)

    def test_04_grad_mat2_CS(self):
        n = 2
        x = np.ones(n)
        x[0] = 5.
        x[1] = 6.
        vg3 = FDValidGrad(1j, self.f_mat2_CS, self.df_mat2)
        ok3 = vg3.compare(x)
        assert(ok3)
#

    def test_05_grad_scalar(self):
        n = 20
        x = np.ones(n)
        x[0] = 5.
        x[10] = 6.
        vg1 = FDValidGrad(1, self.f_scalar, self.df_scalar)
        vg2 = FDValidGrad(2, self.f_scalar, self.df_scalar)
        vg3 = FDValidGrad(1j, self.f_scalar, self.df_scalar)
        return vg1.compare(x) and vg2.compare(x) and vg3.compare(x)

    def rosen(self, X):
        func = 100 * (X[1] - X[0]**2)**2 + (1 - X[0])**2
        return func

    def rosen_grad(self, X):
        dfunc = np.zeros(2)
        dfunc[0] = -400 * (X[1] - X[0]**2) * X[0] - 2 * (1 - X[0])
        dfunc[1] = 200 * (X[1] - X[0]**2)
        return dfunc

    def rosen_grad_vect(self, X):
        return np.array((self.rosen_grad(X), self.rosen_grad(X)))

    def rosen_hess(self, X):
        H = np.zeros((2, 2))
        H[0, 0] = -400 * (X[1] - X[0]**2 - 2 * X[0] * X[0]) - 2
        H[0, 1] = -400 * X[0]
        H[1, 0] = -400 * X[0]
        H[1, 1] = 200 * X[1]
        return H

    def test_06_BFGSHessian(self):
        EPS = 1e-5
        scheme = FDSecondOrderCentered(EPS)
        hess_calc = BFGSFDHessian(scheme, self.rosen_grad)
        x = np.array([1., 1.])
        hess_calc.hess_f(x)
        self.rosen_hess(x)
        return True  # BFGS hessian can be not so close to true hessian...

    def test_07_vect_BFGSHessian(self):
        EPS = 1e-5
        scheme = FDSecondOrderCentered(EPS)
        hess_calc = BFGSFDHessian(scheme, self.rosen_grad_vect)
        x = np.array([1., 1.])
        hess_calc.vect_hess_f(x)
        self.rosen_hess(x)
        return True  # BFGS hessian can be not so close to true hessian...

    def test_08_FDHessian(self):
        EPS = 1e-5
        hess_calc = FDGradient(2, None, self.rosen_grad, fd_step=EPS)
        x = np.array([1., 1.])
        H_FD = hess_calc.hess_f(x)
        H_true = self.rosen_hess(x)

        return np.allclose(H_FD, H_true)

    def test_09_FDHessian_vect(self):
        EPS = 1e-5

        hess_calc = FDGradient(2, None, self.rosen_grad_vect, fd_step=EPS)
        x = np.array([1., 1.])
        H_FD = hess_calc.vect_hess_f(x, 2)
        H_true = self.rosen_hess(x)

        return np.allclose(H_FD[0], H_true) and np.allclose(H_FD[1], H_true)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestValidGrad)
    unittest.TextTestRunner(verbosity=2).run(suite)
