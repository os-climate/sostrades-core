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
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import numpy as np
from sos_trades_core.tools.base_functions.exp_min import compute_func_with_exp_min,\
    compute_dfunc_with_exp_min


class TestBaseFunction(unittest.TestCase):
    """
    Base Function test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        pass

    def test_01_exp_min_function(self):
        '''
        Test the function
        '''
        x = np.linspace(1, -1, 1000)
        min_value = 1.0e-1
        x_minimized = compute_func_with_exp_min(x, min_value)

        # above min value x=x_minimized
        self.assertListEqual(x[x > min_value].tolist(),
                             x_minimized[x > min_value].tolist())
        # under min_value x_minimized is not below 0.9*min_value
        for x_i in x_minimized[x < min_value]:
            self.assertLessEqual(0.9 * min_value, x_i)

    def test_02_exp_min_function_gradient(self):
        '''
        Test the gradient
        '''
        x = np.linspace(1, -1, 100)
        min_value = 1.0e-1
        x_minimized = compute_func_with_exp_min(x, min_value)
        dx = []
        grad = compute_dfunc_with_exp_min(x, min_value)
        for i in range(len(x)):
            x_complex = x.copy()
            # use a step of 1e-8
            x_complex[i] = x[i] + 1e-8
            x_minimized_complex = compute_func_with_exp_min(
                x_complex, min_value)
            # compute gradient using finite differences
            dx.append((x_minimized_complex[i] - x_minimized[i]) / 1e-8)

        self.assertListEqual(np.array(dx).round(6).tolist(),
                             grad.reshape(1, 100)[0].round(6).tolist())
