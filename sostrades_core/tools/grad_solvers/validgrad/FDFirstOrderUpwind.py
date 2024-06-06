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
# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding: iso-8859-1 -*-

from copy import deepcopy

import numpy as np

from .FDScheme import FDScheme


class AbstractFDFirstOrderUpwind(FDScheme):
    """"
    Abstract class for the first order schemes.
    """

    def __init__(self, fd_step, bounds=None):
        FDScheme.__init__(self, fd_step, bounds=bounds)
        self.set_order(1)

    def _generate_x_perturbations(self, e):
        x = self.get_x()
        bounds = self.get_bounds()
        x_pert = []
        for i in range(np.shape(x)[0]):
            x_c = deepcopy(x)
            if bounds is None:
                x_c[i] += e
            else:
                x_c[i] += e * (bounds[i][1] - bounds[i][0])
            x_pert.append(x_c)
        return x_pert


class FDFirstOrderUpwind(AbstractFDFirstOrderUpwind):

    """
    First order upwind finite differences scheme.
    grad = (f(x+fd_step)-f(x))/(fd_step)
    """

    def __init__(self, fd_step, bounds=None):
        AbstractFDFirstOrderUpwind.__init__(self, fd_step, bounds=None)
        self.set_order(1)

    def generate_samples(self):
        x = self.get_x()
        x_samples = [deepcopy(x)]
        e = self.get_fd_step()
        x_samples = x_samples + self._generate_x_perturbations(e)
        self.set_samples(x_samples)

    def compute_grad(self, y_array):
        if isinstance(y_array, type(np.zeros(1))):
            n = len(np.shape(y_array))
            if n < 3:
                grad = (y_array[1:] - y_array[0]).T / self.get_fd_step()
            elif n == 3:
                s = np.shape(y_array)
                p = self.get_grad_dim()
                grad = np.zeros((s[0], s[1], p))
                for i in range(p):
                    grad[:, :, i] = (y_array[:, :, i + 1] -
                                     y_array[:, :, 0]) / self.get_fd_step()
            else:
                raise Exception(
                    "Functional outputs of dimension > 2 are not yet handled")
            return grad
        else:
            raise Exception("FDGradient for functional of type : " +
                            str(type(y_array)) +
                            " not implemented yet.")


class FDFirstOrderUpwindComplexStep(AbstractFDFirstOrderUpwind):

    """
    Complex step, first order gradient calculation.
    Enables a much lower step than real finite differences,
     typically fd_step=1e-30 since there is no
     cancellation error due to a difference calculation

    grad = Imaginary part(f(x+j*fd_step)/(fd_step))
    """

    def __init__(self, fd_step, bounds=None):
        AbstractFDFirstOrderUpwind.__init__(self, fd_step, bounds=None)
        self.set_order(1j)

    def generate_samples(self):
        e = self.get_fd_step() * 1j
        x_samples = self._generate_x_perturbations(e)
        self.set_samples(x_samples)

    def get_x(self):
        return np.array(AbstractFDFirstOrderUpwind.get_x(self), dtype=np.complex128)

    def compute_grad(self, y_array):
        if isinstance(y_array, type(np.zeros(1))):
            n = len(np.shape(y_array))
            if n < 3:
                grad = y_array.T.imag / self.get_fd_step()
            elif n == 3:
                s = np.shape(y_array)
                p = self.get_grad_dim()
                grad = np.zeros((s[0], s[1], p))
                for i in range(p):
                    grad[:, :, i] = y_array[:, :, i].imag / self.get_fd_step()
            else:
                raise Exception(
                    "Functional outputs of dimension > 2 are not yet handled")
            return grad
        else:
            raise Exception("FDGradient for functional of type : " +
                            str(type(y_array)) + " not implemented yet.")
