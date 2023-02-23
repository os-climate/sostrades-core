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
from sostrades_core.execution_engine.proxy_discipline import ProxyDiscipline
'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import pandas as pd
import numpy as np

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestDataIntegrity(unittest.TestCase):
    """
    Scatter data discipline test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.name = 'Coupling'
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        base_path = 'sostrades_core.sos_wrapping.test_discs'
        self.mod_path_all_types = f'{base_path}.disc_all_types.DiscAllTypes'
        self.SUBTYPE = 'subtype_descriptor'

    def test_01_simple_type_vs_value(self):
        '''
        Check the value vs type integrity + if value is None integrity message variable by variable. 
        checked : string,float,array,list,dataframe,dict
        '''
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', self.mod_path_all_types)
        ns_test = self.exec_eng.ns_manager.add_ns(
            'ns_test', self.exec_eng.study_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            disc1_builder)

        self.exec_eng.configure()
        self.exec_eng.set_debug_mode('data_check_integrity')
        wrong_input_dict = {f'{self.exec_eng.study_name}.z': '112',
                            f'{self.exec_eng.study_name}.Disc1.h': [0, 0, 0, 0],
                            f'{self.exec_eng.study_name}.Disc1.dict_in': pd.DataFrame({'key0': [0.] * 3, 'key1': 0.}),
                            f'{self.exec_eng.study_name}.Disc1.df_in': {'key1': 'wrong_type'},
                            f'{self.exec_eng.study_name}.weather': 12}

        input_types = {f'{self.exec_eng.study_name}.z': 'float',
                       f'{self.exec_eng.study_name}.Disc1.h': 'array',
                       f'{self.exec_eng.study_name}.Disc1.dict_in': 'dict',
                       f'{self.exec_eng.study_name}.Disc1.df_in': 'dataframe',
                       f'{self.exec_eng.study_name}.weather': 'string'}

        missing_input_list = [f'{self.exec_eng.study_name}.Disc1.dict_string_in', f'{self.exec_eng.study_name}.Disc1.list_dict_string_in',
                              f'{self.exec_eng.study_name}.Disc1.dict_of_df_in', f'{self.exec_eng.study_name}.Disc1.dict_of_dict_in']
        self.exec_eng.load_study_from_input_dict(wrong_input_dict)

        for data_id, var_data_dict in self.exec_eng.dm.data_dict.items():
            full_name = self.exec_eng.dm.get_var_full_name(data_id)
            if full_name in wrong_input_dict.keys():
                integrity_msg = f"Value {wrong_input_dict[full_name]} has not the type specified in datamanager which is {input_types[full_name]}"
                print(var_data_dict[ProxyDiscipline.CHECK_INTEGRITY_MSG])
                self.assertEqual(
                    var_data_dict[ProxyDiscipline.CHECK_INTEGRITY_MSG], integrity_msg)
            elif full_name in missing_input_list:
                integrity_msg = "Value is not set!"
                print(var_data_dict[ProxyDiscipline.CHECK_INTEGRITY_MSG])
                self.assertEqual(
                    var_data_dict[ProxyDiscipline.CHECK_INTEGRITY_MSG], integrity_msg)
        self.h_data = np.array([0., 0., 0., 0.])
        self.dict_in_data = {'key0': 0., 'key1': 0.}
        self.df_in_data = pd.DataFrame(np.array([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1],
                                                 [0.2, 1.2, 2.2], [-9., -8.7, 1e3]]),
                                       columns=['variable', 'c2', 'c3'])

        self.dict_of_dict_in_data = {'key_A': {'subKey1': 0.1234, 'subKey2': 111.111, 'subKey3': 2036},
                                     'key_B': {'subKey1': 1.2345, 'subKey2': 222.222, 'subKey3': 2036}}
        a_df = pd.DataFrame(np.array([[5., -.05, 5.e5, 5.**5], [2.9, 1., 0., -209.1],
                                      [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                            columns=['key1', 'key2', 'key3', 'key4'])
        self.dict_of_df_in_data = {'key_C': a_df,
                                   'key_D': a_df * 3.1416}

        correct_input_dict = {f'{self.exec_eng.study_name}.z': 112,
                              f'{self.exec_eng.study_name}.Disc1.h': self.h_data,
                              f'{self.exec_eng.study_name}.Disc1.dict_in': {'key0': 0., 'key1': 0.},
                              f'{self.exec_eng.study_name}.Disc1.df_in': self.df_in_data,
                              f'{self.exec_eng.study_name}.weather': 'sunny'}
        self.exec_eng.load_study_from_input_dict(correct_input_dict)
        for data_id, var_data_dict in self.exec_eng.dm.data_dict.items():
            full_name = self.exec_eng.dm.get_var_full_name(data_id)
            if full_name in correct_input_dict.keys():
                integrity_msg = f""
                print(var_data_dict[ProxyDiscipline.CHECK_INTEGRITY_MSG])
                self.assertEqual(
                    var_data_dict[ProxyDiscipline.CHECK_INTEGRITY_MSG], integrity_msg)

        with self.assertRaises(Exception) as cm:
            self.exec_eng.execute()

        not_set_variables_list = [f'{self.exec_eng.study_name}.z_list',
                                  f'{self.exec_eng.study_name}.Disc1.dict_string_in',
                                  f'{self.exec_eng.study_name}.Disc1.list_dict_string_in',
                                  f'{self.exec_eng.study_name}.Disc1.dict_of_dict_in',
                                  f'{self.exec_eng.study_name}.Disc1.dict_of_df_in']
        error_message = '\n'.join(
            [f'Variable {var} : Value is not set!' for var in not_set_variables_list])

        self.assertEqual(str(cm.exception), error_message)

    def test_02_check_range_and_possible_values(self):
        '''
        Check the value range and possible values for :
        1 float
        2 string 
        3 float_list 
        4 string_list. 
        '''
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', self.mod_path_all_types)
        ns_test = self.exec_eng.ns_manager.add_ns(
            'ns_test', self.exec_eng.study_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            disc1_builder)

        self.exec_eng.configure()
        self.exec_eng.set_debug_mode('data_check_integrity')
        wrong_input_dict = {f'{self.exec_eng.study_name}.z': 1e5,
                            f'{self.exec_eng.study_name}.weather': 'cloudy',
                            f'{self.exec_eng.study_name}.z_list': [1e5, 10, -1e5, 0],
                            f'{self.exec_eng.study_name}.weather_list': ['cloudy', 'sunny', 'rainy', 'weather']}

        self.exec_eng.load_study_from_input_dict(wrong_input_dict)

        full_name_z = f'{self.exec_eng.study_name}.z'
        integrity_msg_z = self.exec_eng.dm.get_data(
            full_name_z, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        self.assertEqual(
            integrity_msg_z, f"Value {wrong_input_dict[full_name_z]} is not in range [-10000.0, 10000.0]")

        full_name_z_list = f'{self.exec_eng.study_name}.z_list'
        integrity_msg_z_list = self.exec_eng.dm.get_data(
            full_name_z_list, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        correct_integrity_msg_z_list = f'Value {wrong_input_dict[full_name_z_list]} at index 0 is not in range [-10000.0, 10000.0]'
        correct_integrity_msg_z_list += '\n'
        correct_integrity_msg_z_list += f'Value {wrong_input_dict[full_name_z_list]} at index 2 is not in range [-10000.0, 10000.0]'
        self.assertEqual(
            integrity_msg_z_list, correct_integrity_msg_z_list)

        full_name_weather = f'{self.exec_eng.study_name}.weather'
        integrity_msg_weather = self.exec_eng.dm.get_data(
            full_name_weather, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        self.assertEqual(
            integrity_msg_weather, f"Value {wrong_input_dict[full_name_weather]} not in *possible values* ['cloudy, it is Toulouse ...', 'sunny', 'rainy']")

        full_name_weather_list = f'{self.exec_eng.study_name}.weather_list'
        integrity_msg_weather_list = self.exec_eng.dm.get_data(
            full_name_weather_list, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        correct_integrity_msg_weather_list = f"Value cloudy in list {wrong_input_dict[full_name_weather_list]} not in *possible values* ['cloudy, it is Toulouse ...', 'sunny', 'rainy']"
        correct_integrity_msg_weather_list += '\n'
        correct_integrity_msg_weather_list += f"Value weather in list {wrong_input_dict[full_name_weather_list]} not in *possible values* ['cloudy, it is Toulouse ...', 'sunny', 'rainy']"
        self.assertEqual(
            integrity_msg_weather_list, correct_integrity_msg_weather_list)

    def test_03_check_subtypes_dict_and_list(self):
        '''
        Check the subtypes of different dict and lists
        1 a float_dict
        2 a float list
        3 a string dict list
        4 a df dict
        '''
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', self.mod_path_all_types)
        ns_test = self.exec_eng.ns_manager.add_ns(
            'ns_test', self.exec_eng.study_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            disc1_builder)

        self.exec_eng.configure()
        self.exec_eng.set_debug_mode('data_check_integrity')
        self.dict_of_dict_in_data = {'key_A': {'subKey1': 0.1234, 'subKey2': 111.111, 'subKey3': 2036},
                                     'key_B': {'subKey1': 1.2345, 'subKey2': 222.222, 'subKey3': 2036}}
        a_df = pd.DataFrame(np.array([[5., -.05, 5.e5, 5.**5], [2.9, 1., 0., -209.1],
                                      [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                            columns=['key1', 'key2', 'key3', 'key4'])
        self.dict_of_df_in_data = {'key_C': a_df,
                                   'key_D': a_df * 3.1416}

        wrong_input_dict = {f'{self.exec_eng.study_name}.Disc1.dict_in': {'key0': 1.0, 'key_str': 'wrong type', 'key1': 3, 'key5': {'wrong_dict': 1}},
                            }
        # f'{self.exec_eng.study_name}.Disc1.df_in': pd.DataFrame({'key0': [0.]
        # * 3, 'c2': 0.,'str_df':1.,'c2':4.},
        self.exec_eng.load_study_from_input_dict(wrong_input_dict)

        full_name_dict_in = f'{self.exec_eng.study_name}.Disc1.dict_in'
        integrity_msg_dict_in = self.exec_eng.dm.get_data(
            full_name_dict_in, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        correct_integrity_msg_dict_in = f"Value wrong type in {wrong_input_dict[full_name_dict_in]} should be a float according to subtype descriptor {{'dict': 'float'}}"
        correct_integrity_msg_dict_in += '\n'
        correct_integrity_msg_dict_in += f"Value {wrong_input_dict[full_name_dict_in]['key5']} in {wrong_input_dict[full_name_dict_in]} should be a float according to subtype descriptor {{'dict': 'float'}}"
        self.assertEqual(
            integrity_msg_dict_in, correct_integrity_msg_dict_in)

        correct_input_dict = {f'{self.exec_eng.study_name}.Disc1.dict_in': {'key0': 1.0, 'key_str': 2.0, 'key1': 3, 'key5': 4},
                              }
        # f'{self.exec_eng.study_name}.Disc1.df_in': pd.DataFrame({'key0': [0.]
        # * 3, 'c2': 0.,'str_df':1.,'c2':4.},
        self.exec_eng.load_study_from_input_dict(correct_input_dict)

        full_name_dict_in = f'{self.exec_eng.study_name}.Disc1.dict_in'
        integrity_msg_dict_in = self.exec_eng.dm.get_data(
            full_name_dict_in, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        self.assertEqual(
            integrity_msg_dict_in, '')

        wrong_input_dict2 = {f'{self.exec_eng.study_name}.z_list': [3.0, 'wrong', {'key': 1}]
                             }
        # f'{self.exec_eng.study_name}.Disc1.df_in': pd.DataFrame({'key0': [0.]
        # * 3, 'c2': 0.,'str_df':1.,'c2':4.},
        self.exec_eng.load_study_from_input_dict(wrong_input_dict2)

        full_name_z_list = f'{self.exec_eng.study_name}.z_list'
        integrity_msg_z_list = self.exec_eng.dm.get_data(
            full_name_z_list, ProxyDiscipline.CHECK_INTEGRITY_MSG)

        correct_integrity_msg_z_list = f"Value wrong in {wrong_input_dict2[full_name_z_list]} should be a float according to subtype descriptor {{'list': 'float'}}"
        correct_integrity_msg_z_list += '\n'
        correct_integrity_msg_z_list += f"Value {wrong_input_dict2[full_name_z_list][-1]} in {wrong_input_dict2[full_name_z_list]} should be a float according to subtype descriptor {{'list': 'float'}}"
        correct_integrity_msg_z_list += '\n'
        correct_integrity_msg_z_list += f"Type of wrong ({type('wrong')}) not the same as the type of -10000.0 ({type(-10000.0)}) in range list"
        correct_integrity_msg_z_list += '\n'
        correct_integrity_msg_z_list += f"Type of {wrong_input_dict2[full_name_z_list][-1]} ({type({'key': 1})}) not the same as the type of -10000.0 ({type(-10000.0)}) in range list"
        self.assertEqual(
            integrity_msg_z_list, correct_integrity_msg_z_list)

        wrong_input_dict = {f'{self.exec_eng.study_name}.Disc1.list_dict_string_in': [3.0, {'key': 'str', 'key2': 1}]
                            }
        # f'{self.exec_eng.study_name}.Disc1.df_in': pd.DataFrame({'key0': [0.]
        # * 3, 'c2': 0.,'str_df':1.,'c2':4.},
        self.exec_eng.load_study_from_input_dict(wrong_input_dict)

        full_name = f'{self.exec_eng.study_name}.Disc1.list_dict_string_in'
        integrity_msg = self.exec_eng.dm.get_data(
            full_name, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        correct_integrity_msg = "Value 3.0 should be a dict according to subtype descriptor {'dict': 'string'}"
        correct_integrity_msg += '\n'
        correct_integrity_msg += f"Value 1 in {wrong_input_dict[full_name][-1]} should be a string according to subtype descriptor {{'dict': 'string'}}"
        self.assertEqual(
            integrity_msg, correct_integrity_msg)

        a_df = pd.DataFrame(np.array([[5., -.05, 5.e5, 5.**5], [2.9, 1., 0., -209.1],
                                      [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                            columns=['key1', 'key2', 'key3', 'key4'])
        self.dict_of_df_in_data = {'key_C': a_df,
                                   'key_D': 3.1416}

        wrong_input_dict = {f'{self.exec_eng.study_name}.Disc1.dict_of_df_in': self.dict_of_df_in_data
                            }
        # f'{self.exec_eng.study_name}.Disc1.df_in': pd.DataFrame({'key0': [0.]
        # * 3, 'c2': 0.,'str_df':1.,'c2':4.},
        self.exec_eng.load_study_from_input_dict(wrong_input_dict)

        full_name = f'{self.exec_eng.study_name}.Disc1.dict_of_df_in'
        integrity_msg = self.exec_eng.dm.get_data(
            full_name, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        correct_integrity_msg = f"Value 3.1416 in {wrong_input_dict[full_name]} should be a dataframe according to subtype descriptor {{'dict': 'dataframe'}}"
        self.assertEqual(
            integrity_msg, correct_integrity_msg)

        a_df = pd.DataFrame(np.array([[5., -.05, 5.e5, 5.**5], [2.9, 1., 0., -209.1],
                                      [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                            columns=['key1', 'key2', 'key3', 'key4'])
        self.dict_of_df_in_data = {'key_C': a_df,
                                   'key_D': a_df * 3.1416}

        correct_input_dict = {f'{self.exec_eng.study_name}.Disc1.dict_of_df_in': self.dict_of_df_in_data
                              }
        # f'{self.exec_eng.study_name}.Disc1.df_in': pd.DataFrame({'key0': [0.]
        # * 3, 'c2': 0.,'str_df':1.,'c2':4.},
        self.exec_eng.load_study_from_input_dict(correct_input_dict)

        full_name = f'{self.exec_eng.study_name}.Disc1.dict_of_df_in'
        integrity_msg = self.exec_eng.dm.get_data(
            full_name, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        self.assertEqual(
            integrity_msg, '')

    def test_04_check_dataframe_descriptor(self):
        '''
        Check the subtypes of a dataframe
        '''
        disc1_builder = self.exec_eng.factory.get_builder_from_module(
            'Disc1', self.mod_path_all_types)
        ns_test = self.exec_eng.ns_manager.add_ns(
            'ns_test', self.exec_eng.study_name)
        self.exec_eng.factory.set_builders_to_coupling_builder(
            disc1_builder)

        self.exec_eng.configure()
        self.exec_eng.set_debug_mode('data_check_integrity')
        a_df = pd.DataFrame(np.array([[5., -.05, 5.e5, 5.**5], [2.9, 1., 0., -209.1],
                                      [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                            columns=['variable', 'c2', 'c3', 'key4'])

        wrong_input_dict = {f'{self.exec_eng.study_name}.Disc1.df_in': a_df}

        self.exec_eng.load_study_from_input_dict(wrong_input_dict)

        full_name_dict_in = f'{self.exec_eng.study_name}.Disc1.df_in'
        integrity_msg_dict_in = self.exec_eng.dm.get_data(
            full_name_dict_in, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        correct_integrity_msg_dict_in = f"Dataframe value has a column key4 but the dataframe descriptor has not, df_descriptor keys : dict_keys(['variable', 'c2', 'c3', 'str_df'])"
        self.assertEqual(
            integrity_msg_dict_in, correct_integrity_msg_dict_in)

        a_df = pd.DataFrame(np.array([[5e5, -.05, 5.e5, 5.**5], [2.9, 1., 0., -209.1],
                                      [0.7e-5, 2e3 / 3, 17., 3.1416], [-19., -2., -1e3, 6.6]]),
                            columns=['variable', 'c2', 'c3', 'str_df'])

        a_df = pd.DataFrame({'variable': [5e5, 2.9, 0.7e-5, -19.],
                             'c2': 4,
                             'c3': 8,
                             'str_df': [5.**5, -209.1, 3.1416, 6.6]})
        wrong_input_dict = {f'{self.exec_eng.study_name}.Disc1.df_in': a_df}
        self.exec_eng.load_study_from_input_dict(wrong_input_dict)
        integrity_msg_dict_in = self.exec_eng.dm.get_data(
            full_name_dict_in, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        correct_integrity_msg_dict_in = "Dataframe values in column variable are not in the range [-10000.0, 10000.0] requested in the dataframe descriptor\n"
        correct_integrity_msg_dict_in += "Dataframe values in column str_df are not as type string requested in the dataframe descriptor"

        print(integrity_msg_dict_in)
        self.assertEqual(
            integrity_msg_dict_in, correct_integrity_msg_dict_in)

        a_df_correct = pd.DataFrame({'variable': [5., 2.9, 0.7e-5, -19.],
                                     'c2': 4,
                                     'c3': 8,
                                     'str_df': ['5.**5', '-209.1', '3.1416', '6.6']})
        correct_input_dict = {
            f'{self.exec_eng.study_name}.Disc1.df_in': a_df_correct}
        self.exec_eng.load_study_from_input_dict(correct_input_dict)
        integrity_msg_dict_in = self.exec_eng.dm.get_data(
            full_name_dict_in, ProxyDiscipline.CHECK_INTEGRITY_MSG)
        print(integrity_msg_dict_in)
        self.assertEqual(
            integrity_msg_dict_in, '')


if __name__ == "__main__":
    cls = TestDataIntegrity()
    cls.setUp()
    cls.test_01_simple_type_vs_value()
