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

import re
import numpy as np


from sos_trades_core.tools.cst_manager.constraint_object import ConstraintObject


class ConstraintManager:
    """
    Class to manage constraints for mission/vehicle design
    (if GEMS this class should not be needed anymore)
    """

    def __init__(self):
        """
        Constructor for the Constraint manager class
        """
        self.constraints = {}
        self.main_constraints = []

    def initialize(self):
        """
        Initialize constraints to empty list/dict
        """
        self.constraints = {}
        self.main_constraints = []

    def add_constraint(self, cst_name, key_list, values, weights=None):
        """
        Add a constraint to the dictionary constraints

        :param cst_name: constraint name
        :param key_list: list of key to map the values
        :param values: number or list of number or array of number
        :param weights: same structure as values to specify weights, if omitted considered as 1
        :return: ConstraintObject
        """
        if cst_name not in self.main_constraints:
            self.main_constraints.append(cst_name)

        name = '@'.join([cst_name] + key_list)

        if np.any(np.isnan(values)):
            raise Exception('NaN in {}, values={}'.format(name, values))

        if name in self.constraints:
            raise Exception(
                'Error in add_constraint: {name} is already in the manager'.format(name=name))

        cst_obj = ConstraintObject(values, weights)
        self.constraints[name] = cst_obj

        return cst_obj

    def get_constraint(self, name, cst_func=None, *args, **kwargs):
        """
        Get the constraint with its name
        """
        cst_obj_list = []

        name_list = name.split('@')

        pattern = '(?:@.+)*@'.join(name_list) + '(?:@.+)*'

        for k in self.constraints:
            if re.match(pattern, k):
                cst_obj_list.append(self.constraints[k])

        if cst_obj_list:
            # From the list of constraints extract and concatenate list of
            # values and weights
            values = []
            weights = []
            for obj in cst_obj_list:
                values.append(obj.values.flatten())
                weights.append(obj.weights.flatten())
            values = np.concatenate(values)
            weights = np.concatenate(weights)

            # Apply cst_func
            if cst_func is None:
                cst_func = cst_func_smooth_positive
            result = cst_func(values, weights, *args, **kwargs)

            if np.isnan(result):
                raise Exception('Error in get_constraint for {} with {}: \
                \n\tvalues={}\n\tweights={}\n\tf_args={}\n\tf_kwargs={}'.format(
                    name, cst_func.__name__, values, weights, args, kwargs))

            return result

        else:
            return None

    def export_to_csv(self, filename):
        """
        Export all constraints to a csv file
        """
        strg = ''
        for key, value in self.constraints.items():
            strg += ','.join([key] + [str(x) for x in value.values]) + '\n'

        with open(filename, 'w') as file:
            file.write(strg)


# Functions used in constraint manager (used elsewhere ? )


def cst_func_smooth_positive(values, eps=1e-3, alpha=3):
    """
    Awesome function
    """
    cst_result = np.zeros_like(values)
    for iii, val in enumerate(values):
        if val > 1:
            res = 1 + 2 * np.log(val)
        elif val > eps:
            res = (1. - eps / 2) * val ** 2 + eps * val
        else:
            res = eps * (np.exp(val) - 1.)

        if np.isnan(res):
            print(
                'NaN detected in cst_func_smooth_positive i={}, x={}, r={}'.format(iii, val, res))
        cst_result[iii] = res

    if np.any(np.isnan(cst_result)):
        raise Exception('NaN in cst_func_smooth_positive')

    s_max = smooth_maximum(cst_result, alpha=alpha)

    return s_max


def cst_func_smooth_positive_eq(values, alpha=3):
    """
    Function
    """
    abs_values = np.sign(values) * values
    return cst_func_smooth_positive(abs_values, eps=0., alpha=alpha)


def cst_func_smooth_maximum(values, alpha=3, drop_zeros=False):
    """
    Function
    """
    if drop_zeros:
        val = values[np.where(values)[0]]
    else:
        val = values

    return smooth_maximum(val, alpha=alpha)


def cst_func_hard_max(values):
    """
    Function which return the max of values
    """
    return np.max(values)


def smooth_maximum(cst, alpha=3):
    """
    Function
    """
    max_exp = 650  # max value for exponention input, higher value gives infinity

    max_alphax = np.max(alpha * cst)

    k = max_alphax - max_exp

    den = np.sum(np.exp(alpha * cst - k))
    num = np.sum(cst * np.exp(alpha * cst - k))
    if den != 0:
        result = num / den
    else:
        result = np.max(cst)
        print('Warning in smooth_maximum! den equals 0, hard max is used')

    return result
