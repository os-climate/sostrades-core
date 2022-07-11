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
Exp_min function minimize an array with a min_value with a smooth decreasing exponential 
The gradient of this function can also be used
'''

import numpy as np


def compute_func_with_exp_min(values, min_value):
    '''
    Minimize the values by min_value with an exp function
    If min value is negative we need another function (present in carbonemissiosn model of climateeconomics
    will be soon in this file
    '''
    if min_value < 0:
        raise Exception('The function is not suitable for negative min_value')

    if type(values) != type(np.array([])):
        raise Exception('The function uses np.array as values argument')
    if values.min() < min_value:
        values_copy = values.copy()
        # if some values are below min_value
        # We use the exp smoothing only on values below self.min_value (np.minimum(prod_element, min_value))
        # Then we take the maximum to take prod_element if it is higher
        # than min_value
        # To avoid underflow : exp(-200) is considered to be the
        # minimum value for the exp
        values_copy[values_copy < -200.0 *
                    min_value] = -200.0 * min_value

        min_array = np.ones(len(values_copy)) * min_value
        values_new = np.maximum(
            min_array / 10.0 * (9.0 + np.exp(np.minimum(values_copy, min_array) / min_array)
                                * np.exp(-1)), values_copy)
    else:
        values_new = values.copy()

    return values_new


def compute_dfunc_with_exp_min(values, min_value):

    if min_value < 0:
        raise Exception('The function is not suitable for negative min_value')
    if type(values) != type(np.array([])):
        raise Exception('The function uses np.array as values argument')
    dvalues = np.ones(
        len(values))
    if values.min() < min_value:
        values_copy = values.copy()
            # To avoid underflow : exp(-200) is considered to be the
            # minimum value for the exp
        values_copy[values_copy < -200.0 * min_value] = -200.0 * min_value
        dvalues[values_copy < min_value] = np.exp(
            values_copy[values_copy < min_value] / min_value) * np.exp(-1) / 10.0

    return dvalues.reshape(len(dvalues), 1)
