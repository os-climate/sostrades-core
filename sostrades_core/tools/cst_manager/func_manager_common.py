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

"""
Common file to have methods of func manager (mainly smooth max and it derivative) in other repositories
"""

# pylint: disable=unsubscriptable-object
import numpy as np
from sostrades_core.tools.base_functions.exp_min import compute_dfunc_with_exp_min, compute_func_with_exp_min

def smooth_maximum(cst, alpha=3):
    """
    Function
    """
    max_exp = 650  # max value for exponent input, higher value gives infinity
    min_exp = -300
    max_alphax = np.max(alpha * cst)

    k = max_alphax - max_exp
    # Deal with underflow . max with exp(-300)
    exp_func = np.maximum(min_exp, np.multiply(alpha, cst) - k)
    den = np.sum(np.exp(exp_func))
    num = np.sum(cst * np.exp(exp_func))
    if den != 0:
        result = num / den
    else:
        result = np.max(cst)
        print('Warning in smooth_maximum! den equals 0, hard max is used')

    return result


def smooth_maximum_vect(cst, alpha=3):
    """
    Vectorized version of smooth_maximum function
    """
    cst_array = np.array(cst)
    max_exp = 650  # max value for exponent input, higher value gives infinity
    min_exp = -300
    max_alphax = np.amax(alpha * cst_array, axis=1)

    k = max_alphax - max_exp
    # Deal with underflow . max with exp(-300)
    exp_func = np.maximum(min_exp, alpha * cst_array -
                          np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))
    den = np.sum(np.exp(exp_func), axis=1)
    num = np.sum(cst_array * np.exp(exp_func), axis=1)
    result = np.where(den != 0, num / den, np.amax(cst_array, axis=1))
    if (den == 0).any():
        print('Warning in smooth_maximum! den equals 0, hard max is used')

    return result


def get_dsmooth_dvariable(cst, alpha=3):
    max_exp = 650.0  # max value for exponent input, higher value gives infinity
    min_exp = -300
    alphaxcst = alpha * np.array(cst)
    max_alphax = np.max(alphaxcst)
    #index_max = alphaxcst.index(max_alphax)
    k = max_alphax - max_exp
    exp_func = np.maximum(min_exp, alpha * np.array(cst) - k)
    den = np.sum(np.exp(exp_func))
    num = np.sum(np.array(cst) * np.exp(exp_func))
    d_den = []
    d_num = []
    grad_value = []
    for elem in cst:
        if alpha * elem == max_alphax:
            dden = np.sum([-alpha * np.exp(max(min_exp, alpha * elem_cst - k))
                           for elem_cst in cst if elem_cst * alpha != max_alphax])
            # derivative of den wto cstmax is 0
            dden = dden + 0.0
            d_den.append(dden)
            dnum = np.sum([-alpha * elem_cst * np.exp(max(min_exp, alpha * elem_cst - k))
                           for elem_cst in cst if elem_cst * alpha != max_alphax])
            dnum = dnum + 1.0 * np.exp(alpha * np.array(elem) - k)
            d_num.append(dnum)
            #grad_val_i = dnum / den - (num / den) * (dden / den)
        else:
            exp_func = max(min_exp, alpha * elem - k)
            dden = alpha * np.exp(exp_func)
            d_den.append(dden)
            dnum = elem * (alpha * np.exp(exp_func)
                           ) + np.exp(exp_func)
            d_num.append(dnum)
            # add if den != 0
        grad_val_i = dnum / den - (num / den) * (dden / den)
        grad_value.append(grad_val_i)
    return grad_value


def get_dsmooth_dvariable_vect(cst, alpha=3):
    cst_array = np.array(cst)
    max_exp = 650.0  # max value for exponent input, higher value gives infinity
    min_exp = -300
    alphaxcst = alpha * cst_array

    max_alphax = np.amax(alphaxcst, axis=1)

    k = max_alphax - max_exp
    exp_func = np.maximum(min_exp, alpha * cst_array -
                          np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))
    den = np.sum(np.exp(exp_func), axis=1)
    num = np.sum(cst_array * np.exp(exp_func), axis=1)

    # Vectorized calculation
    exp_func = np.maximum(min_exp, alpha * cst_array -
                          np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))
    dden = alpha * np.exp(exp_func)
    dnum = cst_array * (alpha * np.exp(exp_func)
                        ) + np.exp(exp_func)
    grad_value = dnum / np.repeat(den, cst_array.shape[1]).reshape(cst_array.shape) - (np.repeat(num, cst_array.shape[1]).reshape(
        cst_array.shape) / np.repeat(den, cst_array.shape[1]).reshape(cst_array.shape)) * (dden / np.repeat(den, cst_array.shape[1]).reshape(cst_array.shape))

    # Special case for max element
    max_elem = np.amax(cst_array * np.sign(alpha), axis=1) * np.sign(alpha)
    non_max_idx = np.array([cst_array[i] != max_elem[i]
                            for i in np.arange(cst_array.shape[0])]).reshape(cst_array.shape[0], cst_array.shape[1])
    dden_max = np.sum(-alpha * non_max_idx *
                      np.exp(np.maximum(min_exp, alpha * cst_array - np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))), axis=1)
    dnum_max = np.sum(-alpha * cst_array * non_max_idx *
                      np.exp(np.maximum(min_exp, alpha * cst_array - np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))), axis=1)
    # derivative of den wto cstmax is 0
    dden_max = dden_max + 0.0
    dnum_max = dnum_max + 1.0 * np.exp(alpha * max_elem - k)
    grad_val_max = dnum_max / den - (num / den) * (dden_max / den)

    for i in np.arange(cst_array.shape[0]):
        grad_value[i][np.logical_not(non_max_idx)[i]] = grad_val_max[i]

    return grad_value

def soft_maximum_vect(cst, k=7e2):
    """
    Soft maximum function to get the maximum between array of values while always remaining above or equal to the
    maximum and ensuring gradient continuity.
    The default value of k is intended for arrays scaled between [-1.0, 1.0], the formula will overflow if
    k*max(array)>=710.
    Quasi-arithmetic mean function.
    https://www.johndcook.com/blog/2010/01/13/soft-maximum/
    https://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/
    """
    cst_array = np.array(cst)
    cst_array_limited = np.sign(cst_array)*compute_func_with_exp_min(np.abs(cst_array), 1.0E-15/k)
    if 'complex' in str(cst_array.dtype):
        cst_array_limited += np.imag(cst_array)*1j
    if np.amax(abs(cst_array_limited))*k>709:
        raise ValueError('The absolute value of k*max(cst_array) is too high and would cause a floating point error')
    result = np.log(np.sum(np.exp(k*cst_array_limited), axis=1))/k
    return result

def get_dsoft_maximum_vect(cst, k=7e2):
    """
    Return derivative of soft maximum
    """
    cst_array = np.array(cst)
    cst_array_limited = np.sign(cst_array)*compute_func_with_exp_min(np.abs(cst_array), 1.0E-15/k)
    result = np.log(np.sum(np.exp(k * cst_array_limited), axis=1)) / k

    d_cst_array = np.ones(cst_array.shape)
    d_cst_array_limited = d_cst_array * \
                          compute_dfunc_with_exp_min(np.abs(cst_array), 1.0E-15/k)
    d_exp = k*d_cst_array_limited*np.exp(k*cst_array_limited)
    d_sum = d_exp
    d_log = (1/k) * (d_sum / np.sum(np.exp(k*cst_array_limited), axis=1).reshape(cst_array_limited.shape[0],1))
    d_log = np.where(d_log>1E-20, d_log, 0.0)
    return d_log

def cons_smooth_maximum_vect(cst, alpha=1E16):
    """
    Conservative smooth maximum function.
    This modified version of the smooth_max adds an epsilon value to the smoothed value
    in order to have a conservative value (always smooth_value > max(values))
    """
    cst_array = np.array(cst)
    max_exp = 650  # max value for exponent input, higher value gives infinity
    min_exp = -300
    max_alphax = np.amax(alpha * cst_array, axis=1)

    k = max_alphax - max_exp
    # Deal with underflow . max with exp(-300)
    exp_func = np.maximum(min_exp, alpha * cst_array -
                          np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))
    den = np.sum(np.exp(exp_func), axis=1)
    num = np.sum(cst_array * np.exp(exp_func), axis=1)
    result = np.where(den != 0, num / den + 0.3 / alpha, np.amax(cst_array, axis=1))
    if (den == 0).any():
        print('Warning in smooth_maximum! den equals 0, hard max is used')

    return result

def get_dcons_smooth_dvariable_vect(cst, alpha=1E16):
    cst_array = np.array(cst)
    max_exp = 650.0  # max value for exponent input, higher value gives infinity
    min_exp = -300
    alphaxcst = alpha * cst_array

    max_alphax = np.amax(alphaxcst, axis=1)

    k = max_alphax - max_exp
    exp_func = np.maximum(min_exp, alpha * cst_array -
                          np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))
    den = np.sum(np.exp(exp_func), axis=1)
    num = np.sum(cst_array * np.exp(exp_func), axis=1)

    # Vectorized calculation
    exp_func = np.maximum(min_exp, alpha * cst_array -
                          np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))
    dden = alpha * np.exp(exp_func)
    dnum = cst_array * (alpha * np.exp(exp_func)
                        ) + np.exp(exp_func)
    grad_value = dnum / np.repeat(den, cst_array.shape[1]).reshape(cst_array.shape) - (
                np.repeat(num, cst_array.shape[1]).reshape(
                    cst_array.shape) / np.repeat(den, cst_array.shape[1]).reshape(cst_array.shape)) * (
                             dden / np.repeat(den, cst_array.shape[1]).reshape(cst_array.shape))

    # Special case for max element
    max_elem = np.amax(cst_array * np.sign(alpha), axis=1) * np.sign(alpha)
    non_max_idx = np.array([cst_array[i] != max_elem[i]
                            for i in np.arange(cst_array.shape[0])]).reshape(cst_array.shape[0], cst_array.shape[1])
    dden_max = np.sum(-alpha * non_max_idx *
                      np.exp(np.maximum(min_exp,
                                        alpha * cst_array - np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))),
                      axis=1)
    dnum_max = np.sum(-alpha * cst_array * non_max_idx *
                      np.exp(np.maximum(min_exp,
                                        alpha * cst_array - np.repeat(k, cst_array.shape[1]).reshape(cst_array.shape))),
                      axis=1)
    # derivative of den wto cstmax is 0
    dden_max = dden_max + 0.0
    dnum_max = dnum_max + 1.0 * np.exp(alpha * max_elem - k)
    grad_val_max = dnum_max / den - (num / den) * (dden_max / den)

    for i in np.arange(cst_array.shape[0]):
        grad_value[i][np.logical_not(non_max_idx)[i]] = grad_val_max[i]

    return grad_value