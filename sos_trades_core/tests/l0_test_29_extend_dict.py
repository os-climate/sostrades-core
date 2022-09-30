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

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


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
        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(disc5_builder)

        exec_eng.configure()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.z'] = array([3., 0.])
        values_dict['EE.dict_out'] = {'key1': 0.5, 'key2': 0.5}

        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        target = {
            'EE.z': [
                3.0, 0.0], 'EE.dict_out': [
                0.5, 0.5], 'EE.h': [
                0.75, 0.75]}

        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(list(target[key]), list(res[key]))

    def test_02_sosdiscipline_simple_dict_and_dataframe(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc4_dict_df.Disc4'
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
        rp = exec_eng.root_process.sos_disciplines[0]
        z_out, dict_out = rp.get_sosdisc_outputs(["z", "dict_out"])
        z_out_target = array([0.75, 1.5])
        df_data = {'col1': [1, 2], 'col2': [3, 0.75]}
        df = DataFrame(data=df_data)
        dict_out_target = {'key1': {'key11': 0.75, 'key12': 0.5, 'key13': 8., 'key14': {'key141': df, 'key142': array([5])}},
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

        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc4dict.Disc4'
        disc4_builder = exec_eng.factory.get_builder_from_module(
            'Disc4', mod_list)

        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc5dict.Disc5'
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
        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc5_disc_df.Disc5'
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
        target_localdata = {'EE.dict_out': [0.5, 0.5, 8.0, 0.0, 1.0, 3.0, 1.0, 2.0, 0.5, 5.0, 10.0], 'EE.z': [
            3.0, 0.0], 'EE.h': [0.0, 0.75, 1.0, 0.75]}

        res = {}
        for key in target_localdata:
            res[key] = exec_eng.dm.get_value(key)
            if target_localdata[key] is dict:
                self.assertDictEqual(res[key], target_localdata[key])
            elif target_localdata[key] is array:
                self.assertListEqual(
                    list(target_localdata[key]), list(res[key]))

        # compare output h (sos_trades format) to reference
        rp1 = exec_eng.root_process.sos_disciplines[0]
        h_out = rp1.get_sosdisc_outputs("h")

        h_out_target = {'dataframe': DataFrame(
            data={'col1': array([0.75, 0.75])})}
        self.assertListEqual(
            list(
                h_out.keys()), list(
                h_out_target.keys()), "Incorrect h keys")
        assert_frame_equal(
            h_out['dataframe'],
            h_out_target['dataframe'],
            "Incorrect output 'h' sos_trades format")

    def test_05_extend_dict_soscoupling_nested_dict_and_dataframe(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)

        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc4_dict_df.Disc4'
        disc4_builder = exec_eng.factory.get_builder_from_module(
            'Disc4', mod_list)

        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc5_disc_df.Disc5'
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
        rp1 = exec_eng.root_process.sos_disciplines[1]
        rp0 = exec_eng.root_process.sos_disciplines[0]
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

        # execution
        exec_eng.execute()

        # -- compare outputs (GEMS format) to reference
        target = {'EE.dict_out': [0.707107, 0.5, 8., 0., 1., 3.,
                                  1., 2., 0.707107, 5., 10.],
                  'EE.z': [0.7071067800670185, 1.414213560134037],
                  'EE.h': [0., 0.707107, 1., 0.707107],
                  'EE.Disc4.mydict': array([3., 4.])}

        res = {}
        for key in target:
            res[key] = exec_eng.dm.get_value(key)
            if target[key] is dict:
                self.assertDictEqual(res[key], target[key])
            elif target[key] is array:
                self.assertListEqual(
                    list(target[key]), list(res[key]))

        # -- compare outputs (native format) to reference
        # gather outputs in native format
        dict_out, z_out = rp0.get_sosdisc_outputs(
            ['dict_out', 'z'])
        h_out = rp1.get_sosdisc_outputs('h')
        # compare z
        z_out_target = array([0.70710752, 1.41421503])
        assert_array_almost_equal(
            z_out, z_out_target, decimal=6, err_msg="wrong output z")

        # compare out_dict
        df_data = {'col1': [1, 2], 'col2': [3, 0.7071075153426825]}
        df = DataFrame(data=df_data)
        dict_out_target = {'key1': {'key11': 0.7071075153426825, 'key12': 0.5, 'key13': 8., 'key14': {'key141': df, 'key142': array([5])}},
                           'key2': 10.}
        self.assertSetEqual(set(dict_out.keys()),
                            set(dict_out_target.keys()), "Incorrect dict_out keys")
        self.assertSetEqual(set(dict_out['key1'].keys()),
                            set(dict_out_target['key1'].keys()), "Incorrect dict_out['key1'] keys")
        self.assertSetEqual(set(dict_out['key1']['key14'].keys()),
                            set(dict_out_target['key1']['key14'].keys()), "Incorrect dict_out[key1][key14] keys")
        self.assertAlmostEqual(
            dict_out_target['key1']['key11'],
            dict_out['key1']['key11'], delta=1.0e-6)
        self.assertAlmostEqual(
            dict_out_target['key1']['key12'],
            dict_out['key1']['key12'], delta=1.0e-6)
        self.assertAlmostEqual(
            dict_out_target['key1']['key13'],
            dict_out['key1']['key13'], delta=1.0e-6)
        assert_array_equal(
            dict_out_target['key1']['key14']['key142'],
            dict_out['key1']['key14']['key142'])
        assert_frame_equal(
            dict_out_target['key1']['key14']['key141'],
            dict_out['key1']['key14']['key141'])
        # compare h
        h_out_target = {'dataframe': DataFrame(
            data={'col1': array([0.70710678, 0.70710678])})}
        self.assertListEqual(
            list(
                h_out.keys()), list(
                h_out_target.keys()), "Incorrect h keys")
        assert_frame_equal(
            h_out['dataframe'],
            h_out_target['dataframe'],
            "Incorrect output 'h' sos_trades format")

    def test_06_sosdiscipline_large_dict(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)
        mod_list = 'sos_trades_core.sos_wrapping.test_discs.disc5dict.Disc5'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(disc5_builder)

        exec_eng.configure()

        dict_multi = {f'key{i}': {'1': i + 1, '2': i + 2,
                                  '3': i + 3, '4': i + 4} for i in range(500)}

        values_dict = {f'{self.name}.dict_out': dict_multi,
                       f'{self.name}.z': array([3., 0.])}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.execute()

        # compare GEMS output with reference local data (GEMS format)
        target_localdata = {'EE.dict_out': [0.5, 0.5, 8.0, 0.0, 1.0, 3.0, 1.0, 2.0, 0.5, 5.0, 10.0], 'EE.z': [
            3.0, 0.0], 'EE.h': [0.0, 0.75, 1.0, 0.75]}


if '__main__' == __name__:
    cls = TestExtendDict()
    cls.setUp()
    cls.test_05_extend_dict_soscoupling_nested_dict_and_dataframe()
