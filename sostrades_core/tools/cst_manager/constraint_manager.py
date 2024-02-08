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
from matplotlib import pyplot as plt

from sostrades_core.tools.cst_manager.constraint_object import ConstraintObject
from sostrades_core.tools.base_functions.exp_min import compute_func_with_exp_min, compute_dfunc_with_exp_min
from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager

# pylint: disable=no-value-for-parameter

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
                cst_func = smooth_maximum(FunctionManager.cst_func_ineq())
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

def compute_delta_type(delta, type='abs'):
    if type == 'abs':
        cdelta=np.sqrt(compute_func_with_exp_min(delta ** 2, 1e-15))
    elif type == 'hardmax':
        cdelta = -np.sign(delta) * np.sqrt(compute_func_with_exp_min(delta ** 2, 1e-15))
    elif type == 'hardmin':
        cdelta = np.sign(delta) * np.sqrt(compute_func_with_exp_min(delta ** 2, 1e-15))
    elif type == 'normal':
        cdelta = delta
    else:
        raise Exception('Unknown type of delta_type')
    return cdelta

def compute_dcdelta_dvalue(delta, ddelta_dvalue, type='abs'):
    if type == 'abs':
        dcdelta_dvalue = 2 * delta * ddelta_dvalue * compute_dfunc_with_exp_min(delta ** 2, 1e-15)/(
                    2*np.sqrt(compute_func_with_exp_min(delta ** 2, 1e-15)))
    elif type == 'hardmax':
        dcdelta_dvalue = -np.sign(delta) * 2 * delta * ddelta_dvalue * compute_dfunc_with_exp_min(delta ** 2, 1e-15) / (
                    2 * np.sqrt(compute_func_with_exp_min(delta ** 2, 1e-15)))
    elif type == 'hardmin':
        dcdelta_dvalue = np.sign(delta) * 2 * delta * ddelta_dvalue * compute_dfunc_with_exp_min(delta ** 2, 1e-15) / (
                    2 * np.sqrt(compute_func_with_exp_min(delta ** 2, 1e-15)))
    elif type == 'normal':
        dcdelta_dvalue = ddelta_dvalue
    else:
        raise Exception('Unknown type of delta_type')
    return dcdelta_dvalue

def compute_delta_constraint(value, goal, tolerable_delta=1.0, delta_type='abs', reference_value=1.0, eps=1E-3):
    # Transform inputs into arrays if needed
    length=1
    if isinstance(value, list):
        length = len(value)
    else:
        value=np.array(value)
    if not isinstance(goal, list):
        goal = np.ones(length)*goal
    if not isinstance(tolerable_delta, list):
        tolerable_delta = np.ones(length)*tolerable_delta

    delta = (goal - value)
    cdelta = compute_delta_type(delta, delta_type)
    constraint = ((tolerable_delta - cdelta) / reference_value - eps)
    return constraint

def compute_ddelta_constraint(value, goal, tolerable_delta=1.0, delta_type='abs', reference_value=1.0, eps=1E-3):
    # Transform inputs into arrays if needed
    length = 1
    if isinstance(value, list):
        length = len(value)
    else:
        value = np.array(value)
    if not isinstance(goal, list):
        goal = np.ones(length) * goal
    if not isinstance(tolerable_delta, list):
        tolerable_delta = np.ones(length) * tolerable_delta

    #First step
    delta = (goal - value)
    ddelta_dvalue = -np.identity(len(value))
    ddelta_dgoal = np.identity(len(goal))

    # Second step
    cdelta = compute_delta_type(delta, delta_type)
    dcdelta_dvalue = compute_dcdelta_dvalue(delta, ddelta_dvalue, type=delta_type)
    dcdelta_dgoal = compute_dcdelta_dvalue(delta, ddelta_dgoal, type=delta_type)

    # Third step
    ((tolerable_delta - cdelta) / reference_value - eps)
    ddelta_constraint_dvalue = -dcdelta_dvalue / reference_value
    ddelta_constraint_dgoal = -dcdelta_dgoal / reference_value
    ddelta_constraint_dtolerable_delta = np.identity(len(value)) / reference_value

    return ddelta_constraint_dvalue, ddelta_constraint_dgoal, ddelta_constraint_dtolerable_delta

def delta_constraint_demonstrator():
    """
    Function to plot an example of the delta constraint and its treatment through the function manager
    """
    tolerable_delta = 1.0
    eps=1E-3
    x = np.linspace(-2.0, 2.0, 100)
    y = x
    goal = np.zeros(len(x))
    delta = (goal - y)
    min_valid_x, max_valid_x = x[-1], x[0]
    for i, val in enumerate(delta):
        if np.abs(val)<tolerable_delta:
            min_valid_x=np.minimum(min_valid_x, x[i])
            max_valid_x=np.maximum(max_valid_x, x[i])

    fm=FunctionManager()
    fm.configure_smooth_log(False, 1E20)

    reference_value = 1E3
    delta_type = 'normal'
    constraint = -compute_delta_constraint(y, goal, tolerable_delta=tolerable_delta, delta_type=delta_type, reference_value=reference_value, eps=eps)
    #Multiply constraint by (-1) to mimick the weight that is applied to constraints in func_manager
    cst = FunctionManager.cst_func_eq_lintoquad(fm, -constraint, eps=eps)

    fig, axs = plt.subplots(3)
    axs[0].plot(x, y, label='y')
    axs[0].plot(x, goal, label='goal')
    axs[0].fill_between(x, goal-tolerable_delta, goal+tolerable_delta, label='tolerable_delta', color='green', alpha=0.2)
    axs[0].axvline(min_valid_x, linestyle=':', color='black')
    axs[0].axvline(max_valid_x, linestyle=':', color='black')
    axs[0].legend()
    axs[1].plot(x, constraint*-1.0, label='constraint value * -1')
    axs[1].axhline(eps, linestyle='--', color='red', label='eps')
    axs[1].axvline(min_valid_x, linestyle=':', color='black')
    axs[1].axvline(max_valid_x, linestyle=':', color='black')
    axs[1].legend()
    axs[2].plot(x, cst, '.', label=f'ref={reference_value}')
    axs[2].plot(x,(-constraint) ** 2, linestyle='dotted', color='black')
    axs[2].plot(x, (1. - eps / 2) * (-constraint) ** 2 + eps * -constraint, linestyle='dotted', color='grey')
    axs[2].plot(x, np.abs(eps * (np.exp(-constraint) -1.)), linestyle='dotted', color='green')
    axs[2].axvline(min_valid_x, linestyle=':', color='black')
    axs[2].axvline(max_valid_x, linestyle=':', color='black')
    axs[2].legend()
    plt.show()

def eq_constraint_demonstrator():
    """
    Function to plot an example of the delta constraint and its treatment through the function manager
    """

    eps=1E-3
    x = np.linspace(-2.0, 2.0, 100)
    y = x/1000
    goal = np.zeros(len(x))
    delta = (goal - y)
    min_valid_x, max_valid_x = x[-1], x[0]
    for i, val in enumerate(delta):
        if np.abs(val) < eps:
            min_valid_x = np.minimum(min_valid_x, x[i])
            max_valid_x = np.maximum(max_valid_x, x[i])

    fm=FunctionManager()
    fm.configure_smooth_log(False, 1E20)

    constraint = delta
    #Multiply constraint by (-1) to mimick the weight that is applied to constraints in func_manager
    #cst = FunctionManager.cst_func_eq_delta(fm, constraint*-1, eps=eps)
    cst = FunctionManager.cst_func_eq_lintoquad(fm, constraint * -1, eps=eps)

    fig, axs = plt.subplots(3)
    axs[0].plot(x, y, label='y')
    axs[0].plot(x, goal, label='goal')
    axs[0].legend()
    axs[1].plot(x, constraint*-1.0, label='constraint value * -1')
    axs[1].axhline(eps, linestyle='--', color='red', label='eps')
    axs[1].axhline(-eps, linestyle='--', color='red')
    axs[1].axvline(min_valid_x, linestyle=':', color='black')
    axs[1].axvline(max_valid_x, linestyle=':', color='black')
    axs[1].legend()
    axs[2].plot(x, cst, '.', label=f'cst')
    axs[2].plot(x,(-constraint) ** 2, label='quad', linestyle='dotted', color='red', alpha=0.5)
    axs[2].plot(x, eps * (np.exp(np.abs(constraint)) -1.), label='lin', linestyle='dotted', color='green', alpha=0.5)
    axs[2].axvline(min_valid_x, linestyle=':', color='black')
    axs[2].axvline(max_valid_x, linestyle=':', color='black')
    axs[2].legend()
    plt.show()

def ineq_constraint_demonstrator():
    """
    Function to plot an example of the delta constraint and its treatment through the function manager
    """
    eps = 1E-3
    x = np.linspace(-2.0, 2.0, 100)
    y = x/1000
    goal = np.zeros(len(x))
    delta = (goal - y)
    min_valid_x, max_valid_x = x[-1], x[0]
    for i, val in enumerate(delta):
        if np.abs(val) < eps:
            min_valid_x = np.minimum(min_valid_x, x[i])
            max_valid_x = np.maximum(max_valid_x, x[i])

    self=ConstraintObject
    self.smooth_log=False
    self.eps2=1E20

    constraint = delta
    #Multiply constraint by (-1) to mimick the weight that is applied to constraints in func_manager
    cst=FunctionManager.cst_func_ineq(self, constraint*-1, eps=eps)

    fig, axs = plt.subplots(3)
    axs[0].plot(x, y, label='y')
    axs[0].plot(x, goal, label='goal')
    axs[0].legend()
    axs[1].plot(x, constraint*-1.0, label='constraint value * -1')
    #axs[1].plot(x, constraint_2 * -1.0, label='constraint value 2 * -1')
    axs[1].axhline(eps, linestyle='--', color='red', label='eps')
    axs[1].axvline(min_valid_x, linestyle=':', color='black')
    axs[1].axvline(max_valid_x, linestyle=':', color='black')
    axs[1].legend()
    axs[2].plot(x, cst, '.', label=f'cst')
    axs[2].plot(x,(-constraint) ** 2, label='quad', linestyle='dotted', color='red', alpha=0.5)
    axs[2].plot(x, eps * (np.exp(np.abs(constraint)) -1.), label='lin', linestyle='dotted', color='green', alpha=0.5)
    axs[2].axvline(min_valid_x, linestyle=':', color='black')
    axs[2].axvline(max_valid_x, linestyle=':', color='black')
    axs[2].legend()
    plt.show()