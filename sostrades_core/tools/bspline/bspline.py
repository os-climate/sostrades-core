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
import numpy as np

from scipy.interpolate import BSpline as bspline_sp


class BSpline(object):
    """
    Generic implementation of BSpline.
    """

    def __init__(self, degree=3, n_poles=8, dtype=np.float64, knots=None, errmsg=''):
        self.degree = degree
        self.n_poles = n_poles
        self.knots = None
        self.dtype = dtype
        self.ERROR_MSG = errmsg
        self.set_uniform_knots_list()
        self.ctrl_pts = None
    #-- B-Splines methods

    def set_uniform_knots_list(self):
        """
        set uniform knots_list depending on number of poles
        - n_poles : number of poles
        """
        ERROR_MSG = self.ERROR_MSG + 'set_uniform_knots_list: '
        #-- set knots
        n, n_poles = self.degree, self.n_poles
        m = n_poles + n + 1
        if m < 2 * (n + 1):
            print('n_poles =', n_poles, '\tn =', n)
            raise Exception(ERROR_MSG + ' not enough control points')
        mid_m = m - 2 * (n + 1)

        knots = np.concatenate(
            (np.zeros(n), np.linspace(0, 1, mid_m + 2), np.ones(n)))
        self.knots = knots

    def set_non_uniform_knots_list(self, unif_fact):
        # unif_fact supposed to be in[0;1]
        ERROR_MSG = self.ERROR_MSG + 'set_non_uniform_knots_list: '
        #-- set knots
        n, n_poles = self.degree, self.n_poles
        m = n_poles + n + 1
        if m < 2 * (n + 1):
            print('n_poles =', n_poles, '\tn =', n)
            raise Exception(ERROR_MSG + ' not enough control points')
        mid_m = m - 2 * (n + 1)

        if n_poles == n + 2:
            knots = np.concatenate(
                (np.zeros(n + 1), np.array([unif_fact]), np.ones(n + 1)))  # to be improved
            self.knots = knots
        else:
            print('<!> Warning: forced uniform_knots_list. non_uniform_list function only valid for n_poles = degree+2. Method to be generalized')
            self.set_uniform_knots_list()

    def eval(self, t, diff=0):
        eval_functions = [self.B, self.dBdt, self.d2Bdt2]
        eval_function = eval_functions[diff]
        B_array = np.array([eval_function(t, i)
                            for i in range(self.n_poles)], dtype=self.dtype)
        return B_array

    def float_is_zero(self, v):
        """ Test if floating point number is zero. """
        return v.real == 0.

    def special_div(self, num, den):
        """ Return num/dev with the special rule
        that 0/0 is 0. """
        if self.float_is_zero(num) or self.float_is_zero(den):
            return 0.
        else:
            return num / den

    def B(self, t, i, n=None):
        knots = self.knots
        if n is None:
            n = self.degree
        if n == 0:
            if knots[i].real <= t < knots[i + 1].real:
                return 1.0
            elif i > 0 and knots[i - 1].real < knots[i].real == t == knots[-1].real:
                return 1.0
            else:
                return 0.0
        else:
            left = self.special_div(
                (t - knots[i]) * self.B(t, i, n - 1), knots[i + n] - knots[i])
            right = (1. - self.special_div((t -
                                            knots[i + 1]), knots[i + 1 + n] - knots[i + 1])) * self.B(t, i + 1, n - 1)
            return left + right

    def dBdt(self, t, i, n=None):
        knots = self.knots
        if n is None:
            n = self.degree
        left = self.special_div(n * self.B(t, i, n - 1),
                                knots[i + n] - knots[i])
        right = self.special_div(
            n * self.B(t, i + 1, n - 1), knots[i + 1 + n] - knots[i + 1])
        return left - right

    def d2Bdt2(self, t, i, n=None):
        knots = self.knots
        if n is None:
            n = self.degree
        left = self.special_div(
            n * self.dBdt(t, i, n - 1), knots[i + n] - knots[i])
        right = self.special_div(
            n * self.dBdt(t, i + 1, n - 1), knots[i + 1 + n] - knots[i + 1])
        return left - right

    def set_knots(self, knots):
        self.knots = knots

    def set_ctrl_pts(self, ctrl):
        ERROR_MSG = 'setting control points'
        if len(ctrl) != self.n_poles:
            raise Exception(
                ERROR_MSG + f'control points does not have correct size, must be of length {self.n_poles}')
        self.ctrl_pts = ctrl

    def eval_list_t(self, t_adim):
        """
        Method to evaluate the bspline in an array
        """
        barray_list = np.zeros((len(t_adim), self.n_poles))
        bsp_scipy = bspline_sp(self.knots, self.ctrl_pts, self.degree)
        if isinstance(self.ctrl_pts, list):
            self.ctrl_pts = np.asarray(self.ctrl_pts)

        if self.ctrl_pts.dtype == 'complex128':

            result = np.zeros(len(t_adim), dtype='complex128')

        else:
            result = np.zeros(len(t_adim), dtype=self.dtype)

        #-- compute information
        for i, t in enumerate(t_adim):
            B_array = self.eval(t, diff=0)
            barray_list[i] = B_array
            result[i] = np.dot(self.ctrl_pts, B_array)

            #result[i] = bsp_scipy(t)
        return result, barray_list

    def update_b_array(self, b_array, index_desactivated=None):
        """
        Update b_array to delete element fixed by user
        """
        if index_desactivated is not None:
            updated_barray = []

            for i, k in enumerate(b_array):
                deleted_array = np.delete(k, index_desactivated)
                updated_barray.append(deleted_array.tolist())
        else:
            updated_barray = b_array

        return np.asarray(updated_barray)
