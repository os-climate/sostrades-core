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
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''

import unittest
import pprint
from numpy import array
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from gemseo.utils.compare_data_manager_tooling import dict_are_equal
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_new_type_into_array, \
    convert_array_into_new_type


class TestExtendDict(unittest.TestCase):
    """
    Extend dict type for GEMSEO test class
    """

    def setUp(self):
        self.name = 'EE'
        self.pp = pprint.PrettyPrinter(indent=4, compact=True)

    def test_01_sosdiscipline_simple_dict(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(disc5_builder)

        exec_eng.configure()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.z'] = [3., 0.]
        values_dict['EE.dict_out'] = {'key1': 0.5, 'key2': 0.5}

        exec_eng.load_study_from_input_dict(values_dict)

        disc5 = exec_eng.root_process.proxy_disciplines[0]
        disc5_inputs = {input: exec_eng.dm.get_value(input) for input in disc5.get_input_data_names()}

        target = {
            'EE.z': [3.0, 0.0],
            'EE.dict_out': [0.5, 0.5]}

        converted_inputs = convert_new_type_into_array(disc5_inputs, exec_eng.dm)

        # check new_types conversion into array
        for key, value in target.items():
            self.assertListEqual(value, list(converted_inputs[key]))

        reconverted_inputs = convert_array_into_new_type(converted_inputs, exec_eng.dm)

        # check array conversion into new_types
        self.assertDictEqual(reconverted_inputs['EE.dict_out'], exec_eng.dm.get_value('EE.dict_out'))

    def test_02_sosdiscipline_simple_dict_and_dataframe(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc4_dict_df.Disc4'
        disc4_builder = exec_eng.factory.get_builder_from_module(
            'Disc4', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(disc4_builder)
        exec_eng.configure()
        # -- build input data
        values_dict = {}
        # built my_dict (private in)
        values_dict['EE.Disc4.mydict'] = {'md_1': array([3., 4.])}
        # build dict of dataframe (coupling in)
        h = {'dataframe': DataFrame(data={'col1': array([0.75, 0.75])})}
        values_dict['EE.h'] = h

        # store data
        exec_eng.load_study_from_input_dict(values_dict)

        # -- exec
        exec_eng.execute()

        # compare output h (sos_trades format) to reference
        rp = exec_eng.root_process.proxy_disciplines[0]
        z_out, dict_out = rp.get_sosdisc_outputs(["z", "dict_out"])
        z_out_target = array([0.75, 1.5])
        df_data = {'col1': [1, 2], 'col2': [3, 0.75]}
        df = DataFrame(data=df_data)
        dict_out_target = {
            'key1': {'key11': 0.75, 'key12': 0.5, 'key13': 8., 'key14': {'key141': df, 'key142': array([5])}},
            'key2': 10.}

        assert_array_equal(
            z_out, z_out_target, "wrong output z")

        self.assertSetEqual(set(dict_out.keys()),
                            set(dict_out_target.keys()), "Incorrect dict_out keys")
        self.assertSetEqual(set(dict_out['key1'].keys()),
                            set(dict_out_target['key1'].keys()), "Incorrect dict_out['key1'] keys")
        self.assertSetEqual(set(dict_out['key1']['key14'].keys()),
                            set(dict_out_target['key1']['key14'].keys()), "Incorrect dict_out[key1][key14] keys")
        self.assertAlmostEqual(
            dict_out_target['key1']['key11'],
            dict_out['key1']['key11'])
        self.assertAlmostEqual(
            dict_out_target['key1']['key12'],
            dict_out['key1']['key12'])
        self.assertAlmostEqual(
            dict_out_target['key1']['key13'],
            dict_out['key1']['key13'])
        assert_array_equal(
            dict_out_target['key1']['key14']['key142'],
            dict_out['key1']['key14']['key142'])
        assert_frame_equal(
            dict_out_target['key1']['key14']['key141'],
            dict_out['key1']['key14']['key141'])

    def test_03_soscoupling_simple_dict(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc4dict.Disc4'
        disc4_builder = exec_eng.factory.get_builder_from_module(
            'Disc4', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc4_builder, disc5_builder])
        exec_eng.configure()

        values_dict = {f'{self.name}.dict_out': {'key1': 3., 'key2': 4.},
                       f'{self.name}.z': array([4., 5.]),
                       f'{self.name}.h': array([8., 9.]),
                       f'{self.name}.Disc4.mydict': {'md_1': array([3., 4.])}
                       }
        exec_eng.load_study_from_input_dict(values_dict)

        values_dict_array = {f'{self.name}.dict_out': array([3., 4.]),
                             f'{self.name}.z': array([4., 5.]),
                             f'{self.name}.h': array([8., 9.]),
                             f'{self.name}.Disc4.mydict': array([3., 4.])}

        converted_inputs = convert_new_type_into_array(values_dict, exec_eng.dm)

        # check new_types conversion into array
        self.assertTrue(dict_are_equal(values_dict_array, converted_inputs))

        reconverted_inputs = convert_array_into_new_type(converted_inputs, exec_eng.dm)

        # check array conversion into new_types
        self.assertTrue(dict_are_equal(values_dict, reconverted_inputs))

        exec_eng.execute()

        target = {f'{self.name}.dict_out': {'key1': 0.7071119843035847, 'key2': 0.7071119843035847},
                  f'{self.name}.z': array([0.707111984, 1.41422397]),
                  f'{self.name}.h': array([0.7071067811865475, 0.7071067811865475]),
                  f'{self.name}.Disc4.mydict': {'md_1': array([3., 4.])}}

        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(list(target[key]), list(res[key]))

    def test_04_sosdiscipline_nested_dict(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5_dict_df.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(disc5_builder)

        exec_eng.configure()

        df_data = {'col1': [1, 2], 'col2': [3, 0.5]}
        df = DataFrame(data=df_data)
        values_dict = {f'{self.name}.dict_out': {
            'key1': {'key11': 0.5, 'key12': 0.5, 'key13': 8., 'key14': {'key141': df, 'key142': array([5])}},
            'key2': 10.}, f'{self.name}.z': array([3., 0.])}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        # compare GEMS output with reference local data (GEMS format)
        target_array = {'EE.dict_out': array([0.5, 0.5, 8., 1., 2., 3., 0.5, 5., 10.]), 'EE.z': [3.0, 0.0],
                        'EE.h': array([0.75, 0.75])}
        data_dm = {key: exec_eng.dm.get_value(key) for key in target_array.keys()}
        converted_data_dm = convert_new_type_into_array(data_dm, exec_eng.dm)

        # check new_types conversion into array
        self.assertTrue(dict_are_equal(converted_data_dm, target_array))

        reconverted_data_dm = convert_array_into_new_type(converted_data_dm, exec_eng.dm)

        # check array conversion into new_types
        self.assertTrue(dict_are_equal(data_dm, reconverted_data_dm))

    def test_05_extend_dict_soscoupling_nested_dict_and_dataframe(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc4_dict_df.Disc4'
        disc4_builder = exec_eng.factory.get_builder_from_module(
            'Disc4', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5_dict_df.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc4_builder, disc5_builder])
        exec_eng.configure()

        # fill data in DataManager
        df_data = {'col1': [1, 2], 'col2': [3, 0.5]}
        df = DataFrame(data=df_data)
        values_dict = {}
        dict_out = {
            'key1': {'key11': 0.5, 'key12': 0.5, 'key13': 8., 'key14': {'key141': df, 'key142': array([5])}},
            'key2': 10.}
        values_dict[f'{self.name}.dict_out'] = dict_out
        z = [3., 0.]
        values_dict[f'{self.name}.z'] = z
        values_dict[f'{self.name}.Disc4.mydict'] = {
            'md_1': array([3., 4.])}
        h = {'dataframe': DataFrame(data={'col1': array([0.5, 0.5])})}
        values_dict[f'{self.name}.h'] = h
        exec_eng.load_study_from_input_dict(values_dict)

        # -- compare inputs (GEMS format) to reference
        # gather inputs in native format
        rp1 = exec_eng.root_process.proxy_disciplines[1]
        rp0 = exec_eng.root_process.proxy_disciplines[0]
        dict_out_in, z_in = rp1.get_sosdisc_inputs(['dict_out', 'z'])
        h_in = rp0.get_sosdisc_inputs('h')
        assert_array_almost_equal(
            z, z_in, err_msg="wrong output z")
        assert_frame_equal(
            h['dataframe'], h_in['dataframe'], "wrong output h")
        assert_frame_equal(
            dict_out['key1']['key14']['key141'],
            dict_out_in['key1']['key14']['key141'],
            "wrong output h")

    def test_06_sosdiscipline_large_dict(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(disc5_builder)

        exec_eng.configure()

        dict_multi = {f'key{i}': {'1': i + 1} for i in range(500)}

        values_dict = {f'{self.name}.dict_of_dict_out': dict_multi,
                       f'{self.name}.z': array([3., 0.])}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        # compare GEMS output with reference local data (GEMS format)
        target_array = {'EE.dict_of_dict_out': array([i + 1 for i in range(500)]), 'EE.h': array([0.75, 1.125])}

        data_dm = {key: exec_eng.dm.get_value(key) for key in target_array.keys()}
        converted_data_dm = convert_new_type_into_array(data_dm, exec_eng.dm)

        # check new_types conversion into array
        self.assertTrue(dict_are_equal(converted_data_dm, target_array))

        reconverted_data_dm = convert_array_into_new_type(converted_data_dm, exec_eng.dm)

        # check array conversion into new_types
        self.assertTrue(dict_are_equal(data_dm, reconverted_data_dm))


if '__main__' == __name__:
    cls = TestExtendDict()
    cls.setUp()
    cls.test_05_extend_dict_soscoupling_nested_dict_and_dataframe()
