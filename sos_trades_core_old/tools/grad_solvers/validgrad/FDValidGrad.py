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

from numpy.linalg import norm
from numpy import savetxt
from .FDGradient import FDGradient


class FDValidGrad(object):
    """
    Finite differences gradient calculation and validation.
    """

    def __init__(self, scheme_order, f_pointer, df_pointer, fd_step=1e-6, bounds=None):
        """
        Constructor
        Args:
            scheme_order : order of the numerical scheme : 1, 1j, 2,
            f_pointer : pointer to the function to be derived
            df_pointer : pointer to the function gradient to be checked
            fd_step : finite differences step
        """
        self.__fpointer = f_pointer
        self.__df_pointer = df_pointer
        self.__fd_grad = FDGradient(
            scheme_order, f_pointer, fd_step=fd_step, bounds=bounds)

        self.__multi_proc = False
        self.set_multi_proc(False)

    def set_bounds(self, bounds):
        self.__fd_grad.set_bounds(bounds)

    def set_multi_proc(self, multi):
        self.__multi_proc = multi
        self.__fd_grad.set_multi_proc(multi)

    def compute_fd_grad(self, x, args=None):
        """
        Computes the gradient by finite differences
        Args :
            x : variables where the function is derived
        Returns:
            The gradient vector
        """
        if args is not None:
            return self.__fd_grad.grad_f(x, args)
        return self.__fd_grad.grad_f(x)

    def compare(self, x, treshold=1e-4, args=None, force_print=False, split_out=False, iprint=True, return_all=False, write_to_files=False, grad_files_prefix=""):
        """
        Comparison of provided gradient and finite differences gradient.
        Args :
            x : variables where the function is derived
            treshold : tolerance between analytical and finite differences gradient
            args : function additional args
            force_print : if True, error is printed
            file names of the exported gradient values
            split_out: split checking of vectorial outputs
            iprint : allows printing of messages
            return_all : instead of returning status only, returns status, finite differences gradient and analytical gradients
            write_to_files: write gradients into files
            grad_files_prefix : if write_to_files and gradient is written to disc,

        Returns:
            ok : True if gradient is valid
            df_fd : optional finite differences gradient output
            df: optional analytical gradient output
        """
        df_fd = self.compute_fd_grad(x, args)

        if args is None:
            df = self.__df_pointer(x)
        else:
            df = self.__df_pointer(x, args)

        ok, msg = self.__compute_error_and_check(
            df_fd, df, treshold, split_out=split_out)

        if (not ok or force_print) and iprint:
            print(msg)

        if write_to_files:
            for i in range(len(x)):
                savetxt(grad_files_prefix + 'df_analytic_' +
                        str(i) + '.txt', df[:, :, i].T)
                savetxt(grad_files_prefix + 'df_FD_' +
                        str(i) + '.txt', df_fd[:, :, i].T)

        if return_all:
            return ok, df_fd, df
        else:
            return ok

    def __compute_error_and_check(self, df_fd, df, treshold, split_out=False):
        """
        Computes the relative error between finite differences gradient
        and analytical gradient.
        Args :
            df_fd : the gradient obtained by finite differences
            df : the analytical gradient
            treshold : the numerical tolerance for the comparison
            split_out : option to check each output from a vectorial output

        Returns:
            ok : the status 
            msg : message about the error
        """
        if len(df.shape) == 1 or not split_out:
            nfd = norm(df_fd)
            if nfd < treshold:  # In case df = 0
                err = norm(df_fd - df)
            else:
                err = norm(df_fd - df) / nfd
            if err < treshold:
                ok = True
                msg = 'Gradients are valid.'
            else:
                ok = False
                msg = 'Gradient not in bounds, error = ' + str(err) + '\n'
                msg += 'df =\n' + str(df) + '\n'
                msg += 'df finite diff =\n' + str(df_fd) + '\n'
                msg += 'df-df_fd =\n' + str(df - df_fd) + '\n'
        else:
            ok = True
            err = 0.
            dim_out = df.shape[0]
            err_msg = 'Gradients are not valid due to an error in the output vector\n'
            for n in range(dim_out):
                ndv = len(df_fd[n, :])
                nfd = norm(df_fd[n, :]) / ndv
                if nfd < treshold:  # In case df = 0
                    lerr = norm(df_fd[n, :] - df[n, :])
                else:
                    lerr = norm(df_fd[n, :] - df[n, :]) / nfd
                err += lerr
                if lerr > treshold:
                    ok = False
                    err_msg += 'Error may come from output ' + \
                        str(n) + ' error = ' + str(lerr) + '\n'
            if ok:
                msg = 'Gradients are valid.'
            else:
                msg = err_msg
            err = err / dim_out
        return ok, msg
