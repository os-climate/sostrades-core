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

from copy import deepcopy
import numpy
from numpy.linalg import norm, solve
from scipy.optimize import fsolve

from sos_trades_core.tools.grad_solvers.validgrad.FDValidGrad import FDValidGrad
from sos_trades_core.tools.grad_solvers.validgrad.FDGradient import FDGradient


class NewtonRaphsonProblem():
    """
    Generic implementation of Newton Raphson Problem
    """
    ERROR_MSG = 'ERROR in NewtonRaphsonProblem.'
    WARNING_MSG = 'WARNING in NewtonRaphsonProblem.'

    POSMETH = ['inhouse', 'scipy']

    def __init__(self, W0, R, dRdW, verbose=1):
        """
        Newton-Raphson (Multi-Variate) generic solver
        Documentation to be built from http://fourier.eng.hmc.edu/e161/lectures/ica/node13.html
        """
        # -- Attributes for the Newton-Raphson method
        self.__N = 0
        self.__W0 = None
        self.__W = None

        self.set_W0(W0)

        self.__R = R
        self.__dRdW = dRdW

        self.it = 0

        self.__Res0 = None
        self.__residual = None

        self.__residual_hist = None

        # -- Numerical parameters for the Newton-Raphson method
        self.__method = 'inhouse'
        self.__relax_factor = None
        self.__stop_residual = None
        self.__max_iterations = None
        self.__init_numerical_parameters()

        self.verbose = verbose

        self.bounds = {}

        self.fd_step = 1.e-3

        self.multi_proc = False
        self.fd_mode = 2

    # -- Accessors
    def get_residual_hist(self):
        return self.__residual_hist

    def get_residual(self):
        return self.__residual

    def get_relax_factor(self):
        return self.__relax_factor

    def get_stop_residual(self):
        return self.__stop_residual

    def get_max_iterations(self):
        return self.__max_iterations

    def get_method(self):
        return self.__method

    # -- Numerical parameters related methods
    def __init_numerical_parameters(self):
        """
        Default values for the 
        """
        self.__relax_factor = 0.99
        self.__stop_residual = 1e-6
        self.__max_iterations = 100

    def set_relax_factor(self, relax_factor):
        self.__relax_factor = relax_factor

    def set_stop_residual(self, stop_residual):
        self.__stop_residual = stop_residual

    def set_max_iterations(self, max_iterations):
        self.__max_iterations = max_iterations

    def set_res_0(self, res0):
        self.__Res0 = res0

    def set_fd_mode(self, fd_mode):
        self.fd_mode = fd_mode
#     def set_verbose(self, verbose):
#         self.verbose = verbose

    def set_method(self, method):
        WARNING_MSG = self.WARNING_MSG + 'set_method: '
        if method not in self.POSMETH:
            print(WARNING_MSG + 'method not in ' +
                  str(self.POSMETH) + '. set to inhouse.')
            method = 'inhouse'
        self.__method = method

    def set_fd_step(self, fd_step):
        self.fd_step = fd_step

    # -- Initial state
    def set_W0(self, W0):
        """
        set W0, the initial state variables vector and initialize:
        - N: the dimension of the problem
        - W: the state vector used during the Newton-Raphson iterations
        """
        if type(W0) == type([]):
            W0 = numpy.array(W0)
        self.__N = len(W0)
        self.__W0 = W0

    # -- Newton-Raphson solver related methods
    def __init_iterations(self):
        """
        Initialize attributes before starting the iterations
        """
        self.__W = deepcopy(self.__W0)
        # R0 = self.__R(self.__W0)
        # self.__Res0     = norm(R0)
        self.__residual = 1.0
        self.__residual_hist = []

    def __print_residual(self):
        print("  Iteration= " + str(self.it) +
              ", ||R||/||R0|| = " + str(self.__residual))

    def __NRiteration(self):

        R = self.__R(self.__W)
        if self.__dRdW is None:
            FD_grad = FDGradient(self.fd_mode, self.__R, fd_step=self.fd_step)
            FD_grad.multi_proc = self.multi_proc
            dRdW = FD_grad.grad_f(self.__W)
        else:
            dRdW = self.__dRdW(self.__W)

        if self.verbose > 1:
            print('W=', self.__W)
            print('dRdW = ', dRdW)

        try:
            step = self.get_relax_factor() * -solve(dRdW, R)
        except:
            print('dRdW =', dRdW)
            print('R =', R)
            print('W =', self.__W)
            raise
        self.__W = self.__W + step
        if len(self.bounds.keys()) != 0:
            for k in range(len(self.__W)):
                if self.__W[k] < self.bounds[k][0]:
                    self.__W[k] = self.bounds[k][0]
                if self.__W[k] > self.bounds[k][1]:
                    self.__W[k] = self.bounds[k][1]
        # print 'Wk+1 = ',self.__W

        if self.__Res0 is None:
            self.__Res0 = norm(R)

        # Compute stop criteria
        self.__residual = norm(R) / self.__Res0
        self.__residual_hist.append(self.__residual)

    def solve(self):
        WARNING_MSG = self.WARNING_MSG + 'solve: '
        if self.verbose > 0:
            print('Newton-Raphson (multi-variate) iterations...')
        self.__init_iterations()

        if self.__Res0 == 0.:
            print(WARNING_MSG + 'W0 already solves the system of equations.')
            return self.__W
        else:
            self.it = 0
            if self.verbose > 0:
                print('Method used : ' + str(self.__method))
            if self.__method == 'inhouse':
                self.__solve_inhouse()
            elif self.__method == 'scipy':
                self.__solve_scipy()
            else:
                print(WARNING_MSG + 'unkown method ' + str(self.__method))
        if self.verbose > 0:
            print('Done.')
        return self.__W

    def __solve_inhouse(self):
        while ((self.__residual > self.__stop_residual) and (self.it < self.__max_iterations)):
            if self.verbose > 1:
                print('NR Iteration = ', self.it)
            self.__NRiteration()
            if self.verbose > 0:
                self.__print_residual()
            self.it = self.it + 1
        if self.verbose == 0:
            print("Last iteration :")
            self.__print_residual()
        # Final fun
        self.__R(self.__W)
#         if self.__residual > self.__stop_residual:
#             print 'NR not converged',self.__residual
#

    def __solve_scipy(self):
        W, out, iprint, msg = fsolve(self.__R_scipy, self.__W0, fprime=self.__dRdW_scipy,
                                     full_output=True, factor=100, maxfev=self.__max_iterations)
        self.__dRdW_scipy(W)
        print(msg)
        self.__W = W

    def __R_scipy(self, W):
        R = self.__R(W)
        if self.__Res0 is None:
            self.__Res0 = norm(R)
        self.__residual = norm(R) / self.__Res0
        self.__residual_hist.append(self.__residual)
        self.__print_residual()
        self.it = self.it + 1
        return R

    def __dRdW_scipy(self, W):
        R = self.__R(W)
        if self.__dRdW is None:
            FD_grad = FDGradient(2, self.__R, fd_step=self.fd_step)
            dRdW = FD_grad.grad_f(self.__W)
        else:
            dRdW = self.__dRdW(self.__W)
        self.__residual = norm(R) / self.__Res0
        self.__residual_hist.append(self.__residual)
        self.__print_residual()
        self.it = self.it + 1
        return dRdW

    def export_history(self, filename='residual_history.dat'):
        fid = open(filename, 'w')
        for i, res in enumerate(self.__residual_hist):
            fid.write(str(i) + ' ' + str(res) + '\n')
        fid.close()

    def valid_jacobian(self, W0, fd_step=1.e-3, treshold=1.e-3, iprint=True):
        val_grad = FDValidGrad(2, self.__R, self.__dRdW, fd_step=fd_step)
        ok, df_fd, df = val_grad.compare(
            W0, treshold=treshold, iprint=iprint, return_all=True)

        if iprint:
            for j in range(len(df[:, 0])):
                fid = open('gradient_file' + str(j) + '.dat', 'w')
                for i in range(len(W0)):
                    fid.write(str(i) + ' ' +
                              str(df_fd[j, i]) + ' ' + str(df[j, i]) + '\n')
                fid.close()

            print('\n****************************************************')
            if ok:
                print('  Jacobian is valid.')
            else:
                print('  Jacobian is not valid!')
            print('****************************************************')

        return ok, df_fd, df
