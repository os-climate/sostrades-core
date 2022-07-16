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

from scipy.interpolate import splrep, splev, SmoothBivariateSpline
import numpy as np


def complex_interp1d(x, y, k=3, kind=None, bounds_error=None, fill_value=None, s=0):
    tck = splrep(x, y, k=k, s=s)

    def complex_evaluator(x, epsilon):
        dFdx = splev(x, tck, der=1)
        d2Fdx2 = splev(x, tck, der=2)
        return dFdx * epsilon * 1j - 0.5 * d2Fdx2 * epsilon ** 2

    def evaluator(x):
        isndarray = isinstance(x, np.ndarray)
        if not isndarray:
            x = np.array([x])
        Fx = splev(x.real, tck)

        if (x.imag != 0).any():
            Fx = complex_evaluator(x.real, x.imag) + Fx

        if not isndarray:
            Fx = Fx[0]

        return Fx

    return evaluator


# TODO make compatible with numpy arrays
def complex_interp2d(xlin, ylin, z, kx=3, ky=3, s=None, grid=True, verbose=0):
    if grid:
        x, y = np.meshgrid(xlin, ylin)
    else:
        x = np.array(xlin)
        y = np.array(ylin)
    biv_spline = SmoothBivariateSpline(x.flatten(), y.flatten(), np.array(z).flatten(), kx=kx, ky=ky,
                                       s=s)  # force to cubic either way, else we cant get derivative
    #     ev = interp2d(xlin,ylin,z, kind='cubic')

    xbox = (min(xlin), max(xlin))
    ybox = (min(ylin), max(ylin))

    def complex_evaluator(x, y, epsilon, dx, dy):
        if x < xbox[0] or x > xbox[1]:
            dx = 0
        if y < ybox[0] or y > ybox[1]:
            dy = 0
        if dx == 0 and dy == 0:
            dFdx = 0
            d2Fdx2 = 0
        else:
            dFdx = biv_spline.ev(x, y, dx=dx, dy=dy)
            d2Fdx2 = biv_spline.ev(x, y, dx=2 * dx, dy=2 * dy)

        #         return dFdx * epsilon * 1j - 0.5 * d2Fdx2 * epsilon ** 2
        return dFdx * epsilon * 1j

    def evaluator(x, y):
        Fx = biv_spline(x.real, y.real)
        complex_part = 0

        if x.imag != 0:
            complex_part += complex_evaluator(x.real, y.real, x.imag, 1, 0)
        if y.imag != 0:
            complex_part += complex_evaluator(x.real, y.real, y.imag, 0, 1)

        Fx = complex_part + Fx

        return np.array([Fx[0][0]])

    if verbose > 0:
        eps = 1e-3
        error = 0.
        ref = max(abs(np.array(z).flatten()))
        for xv, yv, zv in zip(x.flatten(), y.flatten(), np.array(z).flatten()):
            error_temp = (abs(zv - evaluator(xv, yv)) / ref)
            #         error_temp = (abs(zv-ev(xv,yv))/ref)
            if error_temp > error:
                error = error_temp
        if error > eps:
            print('\033[1;31mInterpolation relative error: ' + str(error) + '\033[1;m')
    #         raise Exception('Bad interpolation')

    return evaluator