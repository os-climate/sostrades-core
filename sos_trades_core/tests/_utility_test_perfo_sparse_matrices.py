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
import timeit

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import numpy as np
from scipy.sparse import dia_matrix, dok_matrix, csr_matrix, csc_matrix, coo_matrix, identity
from scipy.sparse.lil import lil_matrix


def update_lil_matrix_with_sparse_matrices():
    size_small_sparse = 10
    small_matrix = identity(size_small_sparse)

    residuals = np.linspace(0, 100, 100)
    variables = np.linspace(0, 100, 10)
    len_big_matrix = len(residuals) * len(variables) * size_small_sparse
    big_matrix = lil_matrix((len_big_matrix, len_big_matrix))
    i = 0
    for res in residuals:
        for variable in variables:
            big_matrix[i:i + size_small_sparse, i:i +
                       size_small_sparse] = small_matrix
            i += size_small_sparse


def update_dok_matrix_with_sparse_matrices():
    size_small_sparse = 10
    small_matrix = identity(size_small_sparse)

    residuals = np.linspace(0, 100, 100)
    variables = np.linspace(0, 100, 10)
    len_big_matrix = len(residuals) * len(variables) * size_small_sparse
    big_matrix = dok_matrix((len_big_matrix, len_big_matrix))
    i = 0
    for res in residuals:
        for variable in variables:
            big_matrix[i:i + size_small_sparse, i:i +
                       size_small_sparse] = small_matrix
            i += size_small_sparse


def update_lil_matrix_with_arrays():

    size_small_sparse = 10
    small_matrix = identity(size_small_sparse)

    residuals = np.linspace(0, 100, 100)
    variables = np.linspace(0, 100, 10)
    len_big_matrix = len(residuals) * len(variables) * size_small_sparse
    big_matrix = lil_matrix((len_big_matrix, len_big_matrix))
    i = 0
    for res in residuals:
        for variable in variables:

            small_matrix_array = small_matrix.toarray()
            big_matrix[i:i + size_small_sparse, i:i +
                       size_small_sparse] = small_matrix_array
            i += size_small_sparse
    new_matrix = lil_matrix(big_matrix)


def update_dok_matrix_with_dict():

    size_small_sparse = 10
    small_matrix = identity(size_small_sparse).tocoo()

    residuals = np.linspace(0, 100, 100)
    variables = np.linspace(0, 100, 10)
    len_big_matrix = len(residuals) * len(variables) * size_small_sparse
    big_matrix = dok_matrix((len_big_matrix, len_big_matrix))
    i = 0
    for res in residuals:
        for variable in variables:
            dict.update(big_matrix,
                        {(i + sm_i, i + sm_j): sm_value for sm_i, sm_j, sm_value in zip(small_matrix.row, small_matrix.col, small_matrix.data)})
            i += size_small_sparse


if '__main__' == __name__:

    time_lil = timeit.timeit(
        update_dok_matrix_with_sparse_matrices, number=100)
    print('with dok matrix', time_lil, 's')

    time_lil = timeit.timeit(
        update_lil_matrix_with_sparse_matrices, number=100)
    print('with lil matrix', time_lil, 's')

    time_lil = timeit.timeit(update_lil_matrix_with_arrays, number=100)
    print('with lil matrix and arrays', time_lil, 's')
    time_lil = timeit.timeit(update_dok_matrix_with_dict, number=100)
    print('with dok_matrix and dict via coo', time_lil, 's')
