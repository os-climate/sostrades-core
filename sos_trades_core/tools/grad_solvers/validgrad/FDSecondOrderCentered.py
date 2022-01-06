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

from .FDScheme import FDScheme
from numpy import zeros, shape, array
from copy import deepcopy


class FDSecondOrderCentered(FDScheme):
    """
    Abstract class for the second order centered scheme
    """

    def __init__(self, fd_step, bounds=None):
        """
        Constructor
        """
        FDScheme.__init__(self, fd_step, bounds=bounds)
        self.set_order(2)

    def generate_samples(self):
        """
        Generate samples necessary to compute second order finite differences
        """
        x = self.get_x()
        e = self.get_fd_step()
        x_samples = []
        bounds = self.get_bounds()

        for i in range(self.get_grad_dim()):
            x_c = deepcopy(x)
            if bounds is None:
                x_c[i] += e
            else:
                x_c[i] += e * (bounds[i][1] - bounds[i][0])
            x_samples.append(x_c)

        for i in range(self.get_grad_dim()):
            x_c = deepcopy(x)
            if bounds is None:
                x_c[i] -= e
            else:
                x_c[i] -= e * (bounds[i][1] - bounds[i][0])
            x_samples.append(x_c)

        self.set_samples(x_samples)

    def compute_grad(self, y_array):
        """
        Compute gradient using 2nd order finite differences
        """
        if type(y_array) == type(zeros(1)):
            n = len(shape(y_array))
            p = self.get_grad_dim()
            if n < 3:
                grad = array(y_array[:p] - y_array[p:]).T / \
                    (2. * self.get_fd_step())
            elif n == 3:
                s = shape(y_array)
                grad = zeros((s[0], s[1], p))
                for i in range(p):
                    grad[:, :, i] = (
                        y_array[:, :, i] - y_array[:, :, i + p]) / (2. * self.get_fd_step())
            else:
                raise Exception(
                    "Functional outputs of dimension > 2 are not yet handled")

            return grad
        else:
            raise Exception("FDGradient for functional of type : " +
                            str(type(y_array[0])) + " not implemented yet.")

    def compute_hessian(self, dy_array):
        """
        Compute the Hessian matrix of a scalar function from given function gradients by finite differences
        """
        if type(dy_array) == type(zeros(1)):
            n = len(shape(dy_array))
            if n == 2:
                nb_samples = dy_array.shape[1]
                return (dy_array[:nb_samples, :] - dy_array[nb_samples:, :]) / (2 * self.get_fd_step())
            else:
                raise Exception(
                    "Functional outputs of dimension != 2 are not yet handled")
        else:
            raise Exception("FDGradient for functional of type : " +
                            str(type(dy_array)) + " not implemented yet.")
