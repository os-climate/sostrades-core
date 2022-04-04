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
    data_dict_1 = { 'int': 1,
                    'np_int32': int32(2),
                    'np_int64': int64(3),
                    'float': 2.1,
                    'np_float32': float32(3.2),
                    'np_float64': float64(4.2),
                    'np_complex128': complex128(1 + 2j),
                    'bool': True,
                    'none': None,
                    'string': 'aa',
                    'string_list': ['aa', 'bb'],
                    'string_list_list': [['aa'], ['bb']],
                    'string_dict': {'aa':'bb', 'bb': 'cc'},
                    'array': array([]),
                    'array_order': array([1, 2]),
                    'empty_list': [],
                    'list': [1, 2],
                    'order_list': [1, 2],
                    'float_list': [4., 5.],
                    'empty_dict': {'aa':1, 'bb': 2},
                    'empty_dict2': {},
                    'dict': {'aa':1, 'bb': 2},
                    'dict_list': [{'d1':{}}, {'d2': {'a':0}}],
                    'dataframe_empty': df(),
                    'dataframe': df({'a':[1, 2], 'b':[3, 4]}),
                    'dataframe_col': df({'a':[1, 2], 'b':[3, 4]}),
                    'dataframe_index': df({'a':[1, 2], 'b':[3, 4]}, index=[4, 5]),
                    'dataframe_order': df({'a':[1, 2], 'b':[3, 4]}),
                    'df_list': [df({'a':[1, 2], 'b':[3, 4]}), df({'c':[1, 2], 'd':[3, 4]})],
                     'df_list2': [df({'a':[1, 2], 'b':[3, 4]}), df({'c':[1, 2], 'd':[3, 4]})],
                    'df_list_order': [df({'a':[1, 2], 'b':[3, 4]}), df({'c':[1, 2], 'd':[3, 4]})],
                    'df_dict': {'df1': df({'a':[1, 2], 'b':[3, 4]})},
                    'df_dict2': {'df1': df({'a':[1, 2], 'b':[3, 4]}), 'df2': df({'c':[1, 2], 'd':[3, 4]})}}

    data_dict_2 = { 'int': 11,
                    'np_int32': int32(22),
                    'np_int64': int64(33),
                    'float': 3.1,
                    'np_float32': float32(4.2),
                    'np_float64': float64(5.2),
                    'np_complex128': complex128(1 + 5j),
                    'bool': False,
                    'none': 0,
                    'string': 'bb',
                    'string_list': ['bb', 'aa'],
                    'string_list_list': [['bb'], ['aa']],
                    'string_dict': {'aa':'bb', 'bb': 'bb'},
                    'array': array([0]),
                    'array_order': array([2, 1]),
                    'empty_list': [0],
                    'list': [1, 2, 3],
                    'order_list': [2, 1],
                    'float_list': [4.],
                    'empty_dict': {},
                    'empty_dict2': {'aa':1, 'bb': 2},
                    'dict': {'aa':1, 'bb': 1},
                    'dict_list': [{'d2': {'a':0}}, {'d1':{}}],
                    'dataframe_empty': df({'a': [0, 0]}),
                    'dataframe': df({'a':[3, 4], 'b':[1, 2]}),
                    'dataframe_col': df({'a':[1, 2], 'c':[3, 4]}),
                    'dataframe_index': df({'a':[1, 2], 'b':[3, 4]}),
                    'dataframe_order': df({'b':[3, 4], 'a':[1, 2]}),
                    'df_list': [df({'a':[1, 2], 'b':[3, 4]})],
                    'df_list2': [df({'a':[1, 2], 'b':[3, 4]}), df({'c':[1, 2, 3], 'd':[3, 4, 5]})],
                    'df_list_order': [df({'c':[1, 2], 'd':[3, 4]}), df({'a':[1, 2], 'b':[3, 4]})],
                    'df_dict': {'df1': df({'a':[1, 2], 'b':[3, 4]}), 'df2': df({'c':[1, 2], 'd':[3, 4]})},
                    'df_dict2': {'df1': df({'a':[1, 2], 'b':[3, 4]})}}

    def test_01_compare_dict_method(self):

        diff_dict = {}
        compare_dict(self.data_dict_1,
                     self.data_dict_1, '', diff_dict, df_equals=True)
        self.assertDictEqual(diff_dict, {})
        
        diff_dict = {}
        compare_dict(self.data_dict_2,
                     self.data_dict_2, '', diff_dict, df_equals=True)
        self.assertDictEqual(diff_dict, {})
        
        diff_dict = {}        
        compare_dict(self.data_dict_1,
                     self.data_dict_2, '', diff_dict, df_equals=True)
        self.assertEqual(len(diff_dict), len(self.data_dict_1))
        
