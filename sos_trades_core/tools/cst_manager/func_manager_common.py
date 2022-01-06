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

import numpy as np


def smooth_maximum(cst, alpha=3):
    """
    Function
    """
    max_exp = 650  # max value for exponent input, higher value gives infinity
    min_exp = -300
    max_alphax = np.max(alpha * cst)

    k = max_alphax - max_exp
    # Deal with underflow . max with exp(-300)
    exp_func = np.maximum(min_exp, alpha * cst - k)
    den = np.sum(np.exp(exp_func))
    num = np.sum(cst * np.exp(exp_func))
    if den != 0:
        result = num / den
    else:
        result = np.max(cst)
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
    cst = np.array(cst)
    max_exp = 650.0  # max value for exponent input, higher value gives infinity
    min_exp = -300
    alphaxcst = alpha * cst

    max_alphax = np.amax(alphaxcst, axis=1)

    k = max_alphax - max_exp
    exp_func = np.maximum(min_exp, alpha * cst -
                          np.repeat(k, cst.shape[1]).reshape(cst.shape))
    den = np.sum(np.exp(exp_func), axis=1)
    num = np.sum(cst * np.exp(exp_func), axis=1)

    # Vectorized calculation
    exp_func = np.maximum(min_exp, alpha * cst -
                          np.repeat(k, cst.shape[1]).reshape(cst.shape))
    dden = alpha * np.exp(exp_func)
    dnum = cst * (alpha * np.exp(exp_func)
                  ) + np.exp(exp_func)
    grad_value = dnum / np.repeat(den, cst.shape[1]).reshape(cst.shape) - (np.repeat(num, cst.shape[1]).reshape(
        cst.shape) / np.repeat(den, cst.shape[1]).reshape(cst.shape)) * (dden / np.repeat(den, cst.shape[1]).reshape(cst.shape))

    # Special case for max element
    max_elem = np.amax(cst * np.sign(alpha), axis=1) * np.sign(alpha)
    non_max_idx = np.array([cst[i] != max_elem[i]
                            for i in np.arange(cst.shape[0])]).reshape(cst.shape[0], cst.shape[1])
    dden_max = np.sum(-alpha * non_max_idx *
                      np.exp(np.maximum(min_exp, alpha * cst - np.repeat(k, cst.shape[1]).reshape(cst.shape))), axis=1)
    dnum_max = np.sum(-alpha * cst * non_max_idx *
                      np.exp(np.maximum(min_exp, alpha * cst - np.repeat(k, cst.shape[1]).reshape(cst.shape))), axis=1)
    # derivative of den wto cstmax is 0
    dden_max = dden_max + 0.0
    dnum_max = dnum_max + 1.0 * np.exp(alpha * max_elem - k)
    grad_val_max = dnum_max / den - (num / den) * (dden_max / den)

    for i in np.arange(cst.shape[0]):
        grad_value[i][np.logical_not(non_max_idx)[i]] = grad_val_max[i]

    return grad_value
