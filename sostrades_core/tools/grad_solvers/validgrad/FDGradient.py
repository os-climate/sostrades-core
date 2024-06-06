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

import multiprocessing
from copy import deepcopy

import numpy as np

from .FDFirstOrderUpwind import FDFirstOrderUpwind, FDFirstOrderUpwindComplexStep
from .FDSecondOrderCentered import FDSecondOrderCentered


class FDGradient(object):
    """
    Finite differences gradient.
    Computes the gradient by finite differences for a given scheme order.
    """

    def __init__(self, scheme_order, f_pointer, df_pointer=None, fd_step=1.e-8, bounds=None):
        """
        Constructor.
        Args :
            scheme : the numerical scheme
            f_pointer : the pointer to the function on which
            finite differences are computed.
        """
        self.__scheme_order = scheme_order
        self.fd_step = fd_step
        if scheme_order == 1:
            self.__scheme = FDFirstOrderUpwind(fd_step, bounds)
        elif scheme_order == 1j:
            self.__scheme = FDFirstOrderUpwindComplexStep(fd_step, bounds)
        elif scheme_order == 2:
            self.__scheme = FDSecondOrderCentered(fd_step, bounds)
        else:
            raise Exception(
                "Scheme of order" +
                str(scheme_order) +
                " not available now.")

        self.__fpointer = f_pointer
        self.__dfpointer = df_pointer

        self.multi_proc = False

    def set_bounds(self, bounds):
        self.__scheme.set_bounds(bounds)

    def set_multi_proc(self, multi):
        self.multi_proc = multi

    def get_scheme(self):
        """
        Accessor for the scheme.
        Returns :
            The numerical scheme
        """
        return self.__scheme

    def __worker(self, index, x_in, return_dict):
        out = self.__fpointer(x_in)
        return_dict[index] = out

    def grad_f(self, x, args=None):
        """
        Gradient calculation. Calls the numerical scheme.
        Args:
            x : the variables at which gradient is computed.
        """
        #print('grad_f call')

        self.__scheme.set_x(x)
        self.__scheme.generate_samples()

        samples = self.__scheme.get_samples()
        n_samples = len(samples)
        if self.multi_proc:
            n_procs = multiprocessing.cpu_count()
            print('FDGradient: multi-process grad_f, parallel run on ',
                  n_procs, ' procs.')
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            n = 0
            while n < n_samples:
                if n + n_procs < n_samples:
                    n_subs = n_procs
                else:
                    n_subs = n_samples - n
                jobs = []
                for i in range(n_subs):
                    index = n + i
                    x_in = samples[index]
                    p = multiprocessing.Process(
                        target=self.__worker, args=(index, x_in, return_dict))
                    jobs.append(p)
                    p.start()

                for proc in jobs:
                    proc.join()
                n += n_subs
            y = []
            for i in range(n_samples):
                y.append(return_dict[i])
        else:
            y = []
            for x in samples:
                #print('x =',x)
                if args is None:
                    y.append(deepcopy(self.__fpointer(x)))
                else:
                    y.append(deepcopy(self.__fpointer(x, *args)))
                grad_index = len(y)
                #print('grad index = ',grad_index)

        s = np.shape(y[0])
        if len(s) < 2:
            y_array = np.array(y)
        elif len(s) == 2:
            p = len(y)
            if self.get_scheme().order == 1j:
                y_array = np.zeros((s[0], s[1], p), dtype=np.complex128)
            else:
                y_array = np.zeros((s[0], s[1], p))
            for i in range(p):
                y_array[:, :, i] = y[i]
        else:
            raise Exception(
                "Functional outputs of dimension >2 are not yet handled.")
        return self.__scheme.compute_grad(y_array)

    def hess_f(self, x):
        """
        Hessian computation by finite differences based on numerical scheme provided at construction
        Args:
            x : the variables at which hessian is computed.
        """
        if self.__dfpointer is None:
            raise Exception(
                "Gradient is required to compute finite differences Hessian.")
        self.__scheme.set_x(x)
        self.__scheme.generate_samples()

        dy_array = np.zeros((len(self.__scheme.get_samples()), len(x)))
        for i, x in enumerate(self.__scheme.get_samples()):
            dy_array[i, :] = self.__dfpointer(x)
        return self.__scheme.compute_hessian(dy_array)

    def vect_hess_f(self, x, nb_func):
        """
        Vectorized hessian computation
        Args:
            x : the variables at which hessian is computed.
        """
        if self.__dfpointer is None:
            raise Exception(
                "Gradient is required to compute finite differences Hessian.")
        self.__scheme.set_x(x)
        self.__scheme.generate_samples()

        dy_array_list = np.zeros(
            (nb_func, len(self.__scheme.get_samples()), len(x)))
        for i, x in enumerate(self.__scheme.get_samples()):
            dy_array_list[:, i, :] = self.__dfpointer(x)

        H_list = []
        for f in range(nb_func):
            H_list.append(self.__scheme.compute_hessian(
                dy_array_list[f, :, :]))
        return H_list
