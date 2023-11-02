'''
Copyright 2022 Airbus SAS
Modifications on 2023/08/10-2023/11/02 Copyright 2023 Capgemini

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
mode: python; py-indent-offset: 4; tab-width: 8; coding: utf-8
'''
import numpy as np
from sostrades_core.tools.cst_manager.func_manager_common import smooth_maximum
from sostrades_core.tools.base_functions.exp_min import compute_func_with_exp_min

class FunctionManager:
    """
    Class to manage constraints
    """
    OBJECTIVE = 'objective'
    INEQ_CONSTRAINT = 'ineq_constraint'
    EQ_CONSTRAINT = 'eq_constraint'
    CONSTRAINTS = [INEQ_CONSTRAINT, EQ_CONSTRAINT]
    FTYPE = 'ftype'
    VALUE = 'value'
    WEIGHT = 'weight'  # Can be used for normalisation
    AGGR = 'aggr'
    AGGR_TYPE_SMAX = 'smax'
    AGGR_TYPE_SUM = 'sum'
    AGGR_TYPE_DELTA = 'delta'
    AGGR_TYPE_LIN_TO_QUAD = 'lin_to_quad'
    POS_AGGR_TYPE = [AGGR_TYPE_SMAX, AGGR_TYPE_SUM, AGGR_TYPE_DELTA, AGGR_TYPE_LIN_TO_QUAD]

    def __init__(self):
        """
        Constructor for the function manager class
        -Objective should be normalized (around 1. near optimum)
        -Constraints should be normalized (Around 1. for strong violation)
        -Inequatity constraints: satisfied < 0., violation > 0., strong violation > 1
        -Equality constraints: satisfied: 0.
        """
        self.POS_FTYPE = [self.OBJECTIVE,
                          self.INEQ_CONSTRAINT, self.EQ_CONSTRAINT]
        self.reinit()
        #-- We could use it to log?

    def reinit(self):
        """
        Initialize functions dict
        """
        self.functions = {}
        self.mod_functions = {}
        self.aggregated_functions = {}
        self.mod_obj = 0.
        self.smooth_log = False
        self.eps2 = 1e20

    def configure_smooth_log(self, smooth_log, eps2):
        self.smooth_log = smooth_log
        self.eps2 = eps2

    def __to_array_type(self, value):
        t_val = type(value)
        if t_val == type(np.array([0.])):
            mod_value = value
        elif t_val == type([]):
            mod_value = np.array(value)
        elif t_val == type(0.):
            mod_value = np.array(value)
        else:
            raise ValueError('Unsupported type ' + str(t_val))
        return mod_value

    def add_function(self, tag, value=None, ftype=INEQ_CONSTRAINT, weight=1., aggr_type='sum'):
        """
        By default aggr_type to smax for constraints, thus use keyword for objectives
        """
        if value is not None:
            value = self.__to_array_type(value)
        if aggr_type not in self.POS_AGGR_TYPE:
            raise ValueError(str(aggr_type) + ' not in ' +
                             str(self.POS_AGGR_TYPE))
        dict_func = {}
        dict_func[self.VALUE] = value
        dict_func[self.FTYPE] = ftype
        dict_func[self.WEIGHT] = weight
        dict_func[self.AGGR] = aggr_type
        self.functions[tag] = dict_func

    def update_function_fweight(self, tag, weight):
        self.functions[tag][self.FTYPE] = weight

    def update_function_ftype(self, tag, ftype):
        self.functions[tag][self.FTYPE] = ftype

    def update_function_value(self, tag, value):
        self.functions[tag][self.VALUE] = self.__to_array_type(value)

    def set_aggregation_mods(self, aggr_ineq, aggr_eq):
        self.aggr_mod_ineq = aggr_ineq
        self.aggr_mod_eq = aggr_eq

    def scalarize_all_functions(self, eps=1e-3, alpha=3):
        for tag in self.functions.keys():
            weight = self.functions[tag][self.WEIGHT]
            dict_mod_func = {}
            dict_mod_func[self.FTYPE] = self.functions[tag][self.FTYPE]
            dict_mod_func[self.AGGR] = self.functions[tag][self.AGGR]
            #-- All values are an np array even single values
            #-- Weights are applied here to allow sign modification
            values = weight * self.functions[tag][self.VALUE]
            aggr_type = dict_mod_func[self.AGGR]
            if self.functions[tag][self.FTYPE] == self.OBJECTIVE:
                #-- smooth maximum of values return the value if it was a float
                #-- return smooth maximum if objective was an array
                if aggr_type == 'smax':
                    res = smooth_maximum(values, alpha)
                elif aggr_type == 'sum':
                    res = values.sum()
            elif self.functions[tag][self.FTYPE] == self.INEQ_CONSTRAINT:
                #-- scale between (0., +inf) and take smooth maximum
                cst = self.cst_func_ineq(values, eps, tag)
                res = smooth_maximum(cst, alpha)
            elif self.functions[tag][self.FTYPE] == self.EQ_CONSTRAINT:
                if aggr_type == 'delta':
                    cst = self.cst_func_eq_delta(values, eps, tag)
                elif aggr_type == 'lin_to_quad':
                    cst = self.cst_func_eq_lintoquad(values, eps, tag)
                else:
                    cst = self.cst_func_eq(values)
                res = smooth_maximum(cst, alpha)

            dict_mod_func[self.VALUE] = res
            self.mod_functions[tag] = dict_mod_func

    def build_aggregated_functions(self, eps=1e-3, alpha=3):
        """
        Suppose objectives are scaled between 0. and 1.
        Suppose constraints are scaled also
        need to multiply by 100. to help optimizers numerical instability
        """
        self.scalarize_all_functions(eps, alpha)
        all_mod_obj = []
        all_mod_ineq_cst = []
        all_mod_eq_cst = []
        for tag in self.mod_functions.keys():
            if self.mod_functions[tag][self.FTYPE] == self.OBJECTIVE:
                all_mod_obj.append(self.mod_functions[tag])
            elif self.mod_functions[tag][self.FTYPE] == self.INEQ_CONSTRAINT:
                all_mod_ineq_cst.append(self.mod_functions[tag])
            elif self.mod_functions[tag][self.FTYPE] == self.EQ_CONSTRAINT:
                all_mod_eq_cst.append(self.mod_functions[tag])

        #-- Objective aggregation: sum all the objectives
        self.aggregated_functions[self.OBJECTIVE] = 0.
        for obj_dict in all_mod_obj:
            self.aggregated_functions[self.OBJECTIVE] += obj_dict[self.VALUE]

        #-- Inequality constraint aggregation: takes the smooth maximum
        ineq_cst_val = []
        for ineq_dict in all_mod_ineq_cst:
            ineq_cst_val.append(ineq_dict[self.VALUE])
        ineq_cst_val = np.array(ineq_cst_val)
        if len(ineq_cst_val) > 0:
            if self.aggr_mod_ineq == 'smooth_max':
                self.aggregated_functions[self.INEQ_CONSTRAINT] = self.cst_func_smooth_maximum(
                    ineq_cst_val, alpha)
            else:
                self.aggregated_functions[self.INEQ_CONSTRAINT] = ineq_cst_val.sum()

        else:
            self.aggregated_functions[self.INEQ_CONSTRAINT] = 0.

        #-- Equality constraint aggregation: takes the smooth maximum
        eq_cst_val = []
        for eq_dict in all_mod_eq_cst:
            eq_cst_val.append(eq_dict[self.VALUE])
        eq_cst_val = np.array(eq_cst_val)
        if len(eq_cst_val) > 0:
            if self.aggr_mod_eq == 'smooth_max':
                self.aggregated_functions[self.EQ_CONSTRAINT] = self.cst_func_smooth_maximum(
                    eq_cst_val, alpha)
            else:
                self.aggregated_functions[self.EQ_CONSTRAINT] = eq_cst_val.sum()
        else:
            self.aggregated_functions[self.EQ_CONSTRAINT] = 0.

        #--- Lagrangian objective calculation: sum the aggregated objective and constraints * 100.
        self.mod_obj = 0.
        self.mod_obj += self.aggregated_functions[self.OBJECTIVE]
        self.mod_obj += self.aggregated_functions[self.INEQ_CONSTRAINT]
        self.mod_obj += self.aggregated_functions[self.EQ_CONSTRAINT]
        self.mod_obj = 100. * self.mod_obj
        return self.mod_obj

    def cst_func_eq(self, values, tag='cst'):
        """
        Function
        """
        abs_values = np.sqrt(np.sign(values) * values)
        return self.cst_func_ineq(abs_values, 0., tag=tag)

    def cst_func_eq_delta(self, values, eps=1e-3, tag='cst'):
        """
        Function
        """
        abs_values = np.sqrt(compute_func_with_exp_min(np.array(values) ** 2, 1e-15))
        return self.cst_func_ineq(abs_values, eps, tag=tag)

    def cst_func_ineq(self, values, eps=1e-3, tag='cst'):
        """
        Awesome function
        """
        cst_result = np.zeros_like(values)
        for iii, val in enumerate(values):
            if self.smooth_log and val.real > self.eps2:
                # res0 is the value of the function at val.real=self.eps2 to
                # ensure continuity
                res0 = eps * (np.exp(eps) - 1.)
                res00 = res0 + self.eps2 ** 2 - eps ** 2
                res = res00 + 2 * np.log(val)
                print(
                    f'{tag} = {val.real} > eps2 = {self.eps2}, the log function is applied')
            elif val.real > eps:
                res0 = eps * (np.exp(eps) - 1.)
                res = res0 + val ** 2 - eps ** 2
            elif val.real < -250.0:
                res = 0.0
            else:
                res = eps * (np.exp(val) - 1.)
            if np.isnan(res):
                print(
                    'NaN detected in cst_func_smooth_positive i={}, x={}, r={}, name={}'.format(iii, val, res, tag))
            cst_result[iii] = res

        if np.any(np.isnan(cst_result)):
            raise Exception('NaN in cst_func_smooth_positive {}'.format(tag))

        return cst_result

    def cst_func_eq_lintoquad(self, values, eps=1e-3, tag='cst'):
        """
        Same as cst_func_eq but with a linear increase for negative value
        """
        cst_result = np.zeros_like(values)
        for iii, val in enumerate(values):
            if val.real > eps:
                #if val > eps: quadratic
                res0 = eps * (np.exp(eps) - 1.)
                res = res0 + val ** 2 - eps ** 2
            elif -eps < val.real < 0:
                # if val < 0: linear
                res = eps * (np.exp(-val) - 1.)
            elif val.real < -eps:
                res0 = eps * (np.exp(eps) - 1.)
                res= res0 + (-val) - eps
            else:
                # if 0 < val < eps: linear
                res = eps * (np.exp(val) - 1.)
            if np.isnan(res):
                print(
                    'NaN detected in cst_func_smooth_positive i={}, x={}, r={}, name={}'.format(iii, val, res, tag))
            cst_result[iii] = res

        if np.any(np.isnan(cst_result)):
            raise Exception('NaN in cst_func_smooth_positive {}'.format(tag))

        return cst_result

    def cst_func_smooth_maximum(self, values, alpha=3, drop_zeros=False):
        """
        Function
        """
        if drop_zeros:
            val = values[np.where(values)[0]]
        else:
            val = values

        return smooth_maximum(val, alpha=alpha)

    def get_mod_func_val(self, tag):
        ''' 
        get modified value
        '''
        return self.mod_functions[tag][self.VALUE]

    def get_ineq_constraints_names(self, with_mod_value=False):
        '''
        returns all ineq constraint data
        '''
        def filter_ineq(name):
            return self.functions[name][self.FTYPE] == self.INEQ_CONSTRAINT

        ineq_names = list(filter(filter_ineq, list(self.functions.keys())))

        if not with_mod_value:
            return ineq_names
        else:
            return dict([(n, self.mod_functions[n][self.VALUE]) for n in ineq_names])

    def get_eq_constraints_names(self, with_mod_value=False):
        '''
        returns all eq constraint data
        '''
        functions = self.functions

        def filter_eq(name):
            return functions[name][self.FTYPE] == self.EQ_CONSTRAINT

        eq_names = list(filter(filter_eq, list(functions.keys())))

        if not with_mod_value:
            return eq_names
        else:
            return dict([(n, self.mod_functions[n][self.VALUE]) for n in eq_names])

    def get_constraints_names(self, with_mod_value=False):
        '''
        returns list of all the constraints names, dict name:value
        '''
        eq_data = self.get_eq_constraints_names()
        ineq_data = self.get_ineq_constraints_names()

        if not with_mod_value:
            return list(eq_data.keys()) + list(ineq_data.keys())
        else:
            return {**eq_data, **ineq_data}

    def get_objectives_names(self, with_mod_value=False):
        '''
        returns all objectives data
        '''
        functions = self.functions

        def filter_obj(name):
            return self.functions[name][self.FTYPE] == self.OBJECTIVE

        obj_names = list(filter(filter_obj, list(functions.keys())))

        if not with_mod_value:
            return obj_names
        else:
            return dict([(n, self.mod_functions[n][self.VALUE]) for n in obj_names])

    def compute_dobjective_dweight(self, variable_name):

        return self.functions[variable_name][self.VALUE]

    @staticmethod
    def scale_function(val, val_range):
        """
        Scale a function in a range [a, b] that is mapped to [0, 1]. It can be a<b or a>b.
        :param val: value to scale
        :param val_range: range for the function as [ideal, anti-ideal] (so ideal > anti-ideal for maximisation)
        :return: scaled function with 0 corresponding to ideal value and 1 to anti-ideal
        """
        # TODO: consider using a positive interval and adding a maximisation flag in the sake of clarity
        return np.array([(val - val_range[0]) / (val_range[1] - val_range[0])]).reshape((-1,)) # NB: funcmanager demands arrays of shape (N, )

    @staticmethod
    def scale_function_derivative(val_range) -> float:
        """
        Derivative of the scale function
        :param val_range: range for the function as [ideal, anti-ideal] (so ideal > anti-ideal for maximisation)
        :return: derivative of the scaled function with 0 corresponding to ideal value and 1 to anti-ideal
        """
        return 1. / (val_range[1] - val_range[0])

    @staticmethod
    def unscale_function(val_sc, val_range):
        """
        Unscale a function in a range.
        :param val_sc: scaled function with 0 corresponding to ideal value and 1 to anti-ideal
        :param val_range: range for the function as [ideal, anti-ideal] (so ideal > anti-ideal for maximisation)
        :return: function in original unit
        """
        return val_range[0] + val_sc * (val_range[1] - val_range[0])
