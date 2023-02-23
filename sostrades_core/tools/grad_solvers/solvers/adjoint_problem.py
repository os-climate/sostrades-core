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

import numpy as np
from numpy.linalg import solve


class AdjointProblem(object):
    """
    Generic implementation of Adjoint Problem
    """
    ERROR_MSG = 'ERROR in AdjointProblem.'
    WARNING_MSG = 'WARNING in AdjointProblem.'

    def __init__(self, R, dpRdpW, dpRdpchi, dpFdpW, dpFdpchi):
        """
        Adjoint problem generic solver
        """
        self.__R = R        # Residual of the problem for the resolved state
        self.__dpRdpW = dpRdpW   # Jacobian of the problem for the resolved state
        # partial derivative of the problem with respect to design variables
        # (at the resolved state)
        self.__dpRdpchi = dpRdpchi
        # partial derivative of the cost function with respect to state
        # variable (at the resolved state)
        self.__dpFdpW = dpFdpW
        # partial derivative od the cost function with respect to the design
        # variables (at resolved state)
        self.__dpFdpchi = dpFdpchi

        self.__adjstate = None    # adjoint state vector
        self.__conv_corr = None    # adjoint convergence correction
        # gradient assembly (return dF/dchi total derivative)
        self.__dFdchi = None

    def get_adjoint_state(self):
        return self.__adjstate

    def get_convergence_correction(self):
        return self.__conv_corr

    def get_dFdchi(self):
        return self.__dFdchi

    def solve(self):
        """
        Solve the adjoint problem and compute gradient assembly and adjoint convergence correction
        """
        self.__adjoint()
        self.__convergence_correction()
        self.__adjoint_gradient_assembly()

    def __adjoint(self):
        """
        Compute the adjoint state vector of the problem
        """
        self.__adjstate = solve(self.__dpRdpW.T, -self.__dpFdpW.T)

    def __convergence_correction(self):
        """
        Compute the adjoint convergence correction for the problem
        """
        self.__conv_corr = np.dot(self.__adjstate.T, self.__R)

    def __adjoint_gradient_assembly(self):
        """
        Compute the adjoint gradient assembly to (dF/dchi)
        """
        self.__dFdchi = self.__dpFdpchi + \
            np.dot(self.__adjstate.T, self.__dpRdpchi)
