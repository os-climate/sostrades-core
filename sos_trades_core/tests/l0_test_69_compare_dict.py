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
from gemseo.utils.compare_data_manager_tooling import compare_dict
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import numpy as np
from numpy import int32, int64, float32, float64, complex128, NaN, array
from pandas import DataFrame as df


class TestCompareDict(unittest.TestCase):
    """
    Test of compare_dict method, for every data type
    """
    data_dict_1 = {'int': 1,
                   'np_int32': int32(2),
                   'np_int64': int64(3),
                   'float': 2.1,
                   'np_float32': float32(3.2),
                   'np_float64': float64(4.2),
                   'np_complex128': complex128(1 + 2j),
                   'bool': True,
                   'nan': 1,
                   'None': None,
                   'string': 'aa',
                   'string_list': ['aa', 'bb'],
                   'string_list_list': [['aa'], ['bb']],
                   'string_dict': {'aa':'bb', 'bb': 'cc'},
                   'array': array([]),
                   'list': [1, 2.],
                   'order_list': [1, 2],
                   'int_list': [3, 4],
                   'float_list': [4., 5.],
                   'empty_dict': {'aa':1, 'bb': 2},
                   'empty_dict2': {},
                   'dict': {'aa':1, 'bb': 2},
                   'dict_list': [{'d1':{}, 'd2': {'a':0}}]}
    # 'dataframe': , 'df_list': , 'df_dict': ,
    # check df vide, plein, différence d'index
    
    data_dict_2 = {'int': 11,
                   'np_int32': int32(22),
                   'np_int64': int64(33),
                   'float': 3.1,
                   'np_float32': float32(4.2),
                   'np_float64': float64(5.2),
                   'np_complex128': complex128(1 + 5j),
                   'bool': False,
                   'nan': NaN,
                   'None': 0,
                   'string': 'bb',
                   'string_list': ['bb', 'aa'],
                   'string_list_list': [['bb'], ['aa']],
                   'string_dict': {'aa':'bb', 'bb': 'bb'},
                   'array': array([0]),
                   'list': [1, 2],
                   'order_list': [2, 1],
                   'int_list': [3, 4.],
                   'float_list': [4.],
                   'empty_dict': {},
                   'empty_dict2': {'aa':1, 'bb': 2},
                   'dict': {'aa':1, 'bb': 1},
                   'dict_list': [{'d2': {'a':0}, 'd1':{}}]}

    def test_01_compare_dict_method(self):

        diff_dict = {}
        compare_dict(self.data_dict_1,
                     self.data_dict_1, '', diff_dict, df_equals=True)
        self.assertDictEqual(diff_dict, {})
        
        diff_dict = {}
        compare_dict(self.data_dict_1,
                     self.data_dict_1, '', diff_dict, df_equals=True)
        self.assertDictEqual(diff_dict, {})
        
        diff_dict = {}        
        compare_dict(self.data_dict_1,
                     self.data_dict_2, '', diff_dict, df_equals=True)
        for key in diff_dict.keys():
            print(key)
        self.assertEqual(len(diff_dict), len(self.data_dict_1))
        
        # COMPARAISONS MANQUANTES: 
        # None
        # NaN + dépend de l'ordre des dicos à comparer
        # array vide
        # list, int_list, float_list avec des valeurs identiques mais des types différents 1/1.
        # dict_list ne vérifie pas l'ordre des éléments dans la liste

