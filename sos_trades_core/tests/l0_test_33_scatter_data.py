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
import pandas as pd

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class TestScatterData(unittest.TestCase):
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
        base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.mod1_path = f'{base_path}.disc1_dict.Disc1'
        self.mod2_path = f'{base_path}.disc2.Disc2'
        self.mod2_scatter_gather = f'{base_path}.disc_scatter_gather_data.Disc2'
        self.SUBTYPE = 'subtype_descriptor'

    def test_01_scatter_data(self):
        ns_dict = {'ns_ac': self.namespace}

        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',

                        'input_ns': 'ns_barrierr',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_ac'}

        mydict_y = {'input_name': 'y_dict',
                    'input_type': 'dict',
                    'input_ns': 'ns_ac',
                    'output_name': 'y',
                    'output_type': 'float',
                    'scatter_var_name': 'name_list'}

        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        self.exec_eng.smaps_manager.add_data_map('y_dict', mydict_y)

        disc1 = self.factory.get_builder_from_module('Disc1', self.mod1_path)

        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        scatter_build = self.exec_eng.factory.create_scatter_builder(
            'Disc2', 'name_list', disc2_builder)

        scatter_data = self.exec_eng.factory.create_scatter_data_builder(
            'scatter_data', 'y_dict')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1, scatter_build, scatter_data])

        self.exec_eng.configure()

        dict_values = {self.study_name + '.name_list': ['name_1', 'name_2']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name + '.x_list': [2, 4],
                       self.study_name + '.Disc1.a': 3,
                       self.study_name + '.Disc1.b': 4,
                       self.study_name + '.Disc2.name_1.constant': 10,
                       self.study_name + '.Disc2.name_1.power': 2,
                       self.study_name + '.Disc2.name_2.constant': 20,
                       self.study_name + '.Disc2.name_2.power': 4}
        self.exec_eng.dm.set_values_from_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        # test dm.data_dict content and data_in/data_out referencing in dm

        scatter_data_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.scatter_data')[0]
        disc1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[0]
        disc2 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc2')[0]

        self.assertListEqual(
            [key for key in list(scatter_data_disc._data_in.keys()) if key not in scatter_data_disc.NUM_DESC_IN], [
                'y_dict', 'name_list'])
        self.assertListEqual(list(scatter_data_disc._data_out.keys()), [
            'name_1.y', 'name_2.y'])

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_list'), ['name_1', 'name_2'])
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_list'), scatter_data_disc.get_data_in()['name_list']['value'])
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_list'), disc2.get_data_in()['name_list'])

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.y_dict'), {'name_1': 10, 'name_2': 16})
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.y_dict'), disc1.get_data_out()['y_dict'])

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_1.y'), 10)
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_1.y'), scatter_data_disc.get_data_out()['name_1.y'])

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_2.y'), 16)
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_2.y'), scatter_data_disc.get_data_out()['name_2.y'])

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.t_dict'), {'name_1': 4, 'name_2': 5})

        # test configure/run after name_list modification

        dict_values = {self.study_name + '.name_list': ['name_1']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()

        # test scatter data cleaning after configure
        self.assertNotIn('MyCase.name_2.y',
                         self.exec_eng.dm.data_id_map.keys())

        # check referencing of scatter_data output
        disc2 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc2.name_1')[
            0]
        scatter_data = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.scatter_data')[0]
        full_name_y = 'MyCase.name_1.y'
        dm_referencing = self.exec_eng.dm.get_data(full_name_y)
        self.assertEqual(disc2.get_var_full_name(
            'y', disc2._data_in), full_name_y)
        self.assertEqual(scatter_data.get_var_full_name(
            'name_1.y', scatter_data._data_out), full_name_y)

        # TO DO: correct referencing of disc2 input
        # self.assertTrue(disc2._data_in['y'] is dm_referencing)
        self.assertTrue(scatter_data._data_out['name_1.y'] is dm_referencing)

    def test_02_scatter_data_with_lists_in_map(self):
        ns_dict = {'ns_ac': self.namespace,
                   'ns_data_out': self.namespace}

        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',

                        'input_ns': 'ns_barrierr',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_ac'}

        mydict_data = {'input_name': ['y_dict', 't_dict'],
                       'input_type': ['dict', 'dict'],
                       'input_ns': 'ns_ac',
                       'output_name': ['y', 't'],
                       'output_ns': 'ns_data_out',
                       'output_type': ['float', 'float'],
                       'scatter_var_name': 'name_list'}

        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        self.exec_eng.smaps_manager.add_data_map('data_map', mydict_data)

        disc1 = self.factory.get_builder_from_module('Disc1', self.mod1_path)

        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        scatter_build = self.exec_eng.factory.create_scatter_builder(
            'Disc2', 'name_list', disc2_builder)

        scatter_data = self.exec_eng.factory.create_scatter_data_builder(
            'scatter_data', 'data_map')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1, scatter_build, scatter_data])

        self.exec_eng.configure()

        dict_values = {self.study_name + '.name_list': ['name_1', 'name_2']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()

        # User fill in the fields in the GUI
        dict_values = {self.study_name + '.x_list': [2, 4],
                       self.study_name + '.Disc1.a': 3,
                       self.study_name + '.Disc1.b': 4,
                       self.study_name + '.Disc2.name_1.constant': 10,
                       self.study_name + '.Disc2.name_1.power': 2,
                       self.study_name + '.Disc2.name_2.constant': 20,
                       self.study_name + '.Disc2.name_2.power': 4}
        self.exec_eng.dm.set_values_from_dict(dict_values)

        self.exec_eng.display_treeview_nodes()

        self.exec_eng.execute()

        # test dm.data_dict content and data_in/data_out referencing in dm

        scatter_data_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.scatter_data')[0]
        disc1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[0]

        self.assertListEqual(
            [key for key in list(scatter_data_disc._data_in.keys()) if key not in scatter_data_disc.NUM_DESC_IN], [
                'y_dict', 't_dict', 'name_list'])
        self.assertListEqual(list(scatter_data_disc._data_out.keys()), [
            'name_1.y', 'name_1.t', 'name_2.y', 'name_2.t'])

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.y_dict'), {'name_1': 10, 'name_2': 16})
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_1.y'), 10)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_2.y'), 16)

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.t_dict'), {'name_1': 4, 'name_2': 5})
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_1.t'), 4)
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_2.t'), 5)

    def test_03_scatter_data_with_dataframe(self):
        self.mod2_path = 'sos_trades_core.sos_wrapping.test_discs.disc6.Disc6'
        ns_dict = {'ns_ac': self.namespace,
                   'ns_protected': f'{self.namespace}.Disc2'}

        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',

                        'input_ns': 'ns_barrierr',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_protected',
                        'ns_to_update': ['ns_protected']
                        }

        mydict_y = {'input_name': 'df_full',
                    'input_type': 'dataframe',
                    'input_ns': 'ns_protected',
                    'output_name': 'df',
                    'output_type': 'dataframe',
                    'scatter_var_name': 'name_list',
                    'scatter_column_name': 'name'
                    }

        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        self.exec_eng.smaps_manager.add_data_map('df_full', mydict_y)

        disc1 = self.factory.get_builder_from_module('Disc1', self.mod1_path)

        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        scatter_build = self.exec_eng.factory.create_scatter_builder(
            'Disc2', 'name_list', disc2_builder)

        scatter_data = self.exec_eng.factory.create_scatter_data_builder(
            'scatter_data', 'df_full')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1, scatter_build, scatter_data])

        self.exec_eng.configure()

        dict_values = {self.study_name + '.name_list': ['name_1', 'name_2']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()
        df_default = pd.DataFrame({
            'name': ['name_1', 'name_1', 'name_2', 'name_2'],
            'c1': [1.0, 1.0, 10.0, 10.0],
            'c2': [5.0, 5.0, 50.0, 50.0],
        })
        # User fill in the fields in the GUI
        dict_values = {
            self.study_name + '.x_list': [2, 4],
            self.study_name + '.Disc1.a': 3,
            self.study_name + '.Disc1.b': 4,
            self.study_name + '.Disc2.name_2.dict_df': {'a': df_default, 'b': df_default},
            self.study_name + '.Disc2.name_1.dict_df': {'a': df_default, 'b': df_default},
            self.study_name + '.Disc2.df_full': df_default,
        }
        self.exec_eng.dm.set_values_from_dict(dict_values)

        self.exec_eng.display_treeview_nodes(True)

        self.exec_eng.execute()

        # test dm.data_dict content and data_in/data_out referencing in dm

        scatter_data_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.scatter_data')[0]
        disc1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[0]
        disc2 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc2')[0]

        self.assertListEqual(
            [key for key in list(scatter_data_disc._data_in.keys()) if key not in scatter_data_disc.NUM_DESC_IN], [
                'df_full', 'name_list'])
        self.assertListEqual(list(scatter_data_disc._data_out.keys()), [
            'name_1.df', 'name_2.df'])

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_list'), ['name_1', 'name_2'])
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_list'), scatter_data_disc.get_data_in()['name_list']['value'])
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_list'), disc2.get_data_in()['name_list'])

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.y_dict'), {'name_1': 10, 'name_2': 16})
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.y_dict'), disc1.get_data_out()['y_dict'])

        pd.testing.assert_frame_equal(self.exec_eng.dm.get_value(
            'MyCase.Disc2.name_1.df'), pd.DataFrame({
                'name': ['name_1', 'name_1'],
                'c1': [1.0, 1.0],
                'c2': [5.0, 5.0],
            }))

        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.Disc2.name_1.df'), scatter_data_disc.get_data_out()['name_1.df'])

        pd.testing.assert_frame_equal(self.exec_eng.dm.get_value(
            'MyCase.Disc2.name_2.df'), pd.DataFrame({
                'name': ['name_2', 'name_2'],
                'c1': [10.0, 10.0],
                'c2': [50.0, 50.0],
            }))
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.Disc2.name_2.df'), scatter_data_disc.get_data_out()['name_2.df'])

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.t_dict'), {'name_1': 4, 'name_2': 5})

        # test configure/run after name_list modification

        dict_values = {self.study_name + '.name_list': ['name_1']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()

        # test scatter data cleaning after configure
        self.assertNotIn('MyCase.Disc2.name_2.df',
                         self.exec_eng.dm.data_id_map.keys())

        # check referencing of scatter_data output
        disc2 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc2.name_1')[
            0]
        scatter_data = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.scatter_data')[0]
        full_name_df = 'MyCase.Disc2.name_1.df'
        dm_referencing = self.exec_eng.dm.get_data(full_name_df)
        self.assertEqual(disc2.get_var_full_name(
            'df', disc2._data_in), full_name_df)
        self.assertEqual(scatter_data.get_var_full_name(
            'name_1.df', scatter_data._data_out), full_name_df)

        # TO DO: correct referencing of disc2 input
        # self.assertTrue(disc2._data_in['y'] is dm_referencing)
        self.assertTrue(scatter_data._data_out['name_1.df'] is dm_referencing)

    def test_04_scatter_data_list_with_dataframe(self):
        self.mod2_path = 'sos_trades_core.sos_wrapping.test_discs.disc6.Disc6'
        ns_dict = {'ns_ac': self.namespace,
                   'ns_protected': f'{self.namespace}.Disc2'}

        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',

                        'input_ns': 'ns_barrierr',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_protected',
                        'ns_to_update': ['ns_protected']
                        }

        mydict_y = {'input_name': ['df_dict_dict', 'df_full'],
                    'input_type': ['dict', 'dataframe'],
                    'input_ns': 'ns_protected',
                    'output_name': ['dict_df', 'df'],
                    'output_type': ['dict', 'dataframe'],
                    'scatter_var_name': 'name_list',
                    'scatter_column_name': [None, 'name']
                    }

        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        self.exec_eng.smaps_manager.add_data_map('df_full', mydict_y)

        disc1 = self.factory.get_builder_from_module('Disc1', self.mod1_path)

        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_path)

        scatter_build = self.exec_eng.factory.create_scatter_builder(
            'Disc2', 'name_list', disc2_builder)

        scatter_data = self.exec_eng.factory.create_scatter_data_builder(
            'scatter_data', 'df_full')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [disc1, scatter_build, scatter_data])

        self.exec_eng.configure()

        dict_values = {self.study_name + '.name_list': ['name_1', 'name_2']}
        self.exec_eng.dm.set_values_from_dict(dict_values)
        self.exec_eng.configure()
        df_default = pd.DataFrame({
            'name': ['name_1', 'name_1', 'name_2', 'name_2'],
            'c1': [1.0, 1.0, 10.0, 10.0],
            'c2': [5.0, 5.0, 50.0, 50.0],
        })
        # User fill in the fields in the GUI
        dict_values = {
            self.study_name + '.x_list': [2, 4],
            self.study_name + '.Disc1.a': 3,
            self.study_name + '.Disc1.b': 4,
            self.study_name + '.Disc2.df_dict_dict': {'name_1': {'a': df_default, 'b': df_default},
                                                      'name_2': {'a': df_default, 'b': df_default}},
            self.study_name + '.Disc2.df_full': df_default,
        }
        self.exec_eng.dm.set_values_from_dict(dict_values)

        self.exec_eng.display_treeview_nodes(True)

        self.exec_eng.execute()

        # test dm.data_dict content and data_in/data_out referencing in dm

        scatter_data_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.scatter_data')[0]
        disc1 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc1')[0]
        disc2 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc2')[0]

        self.assertListEqual(
            [key for key in list(scatter_data_disc._data_in.keys()) if key not in scatter_data_disc.NUM_DESC_IN], [
                'df_dict_dict', 'df_full', 'name_list'])
        self.assertListEqual(list(scatter_data_disc._data_out.keys()), [
            'name_1.dict_df', 'name_1.df', 'name_2.dict_df', 'name_2.df'])

        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_list'), ['name_1', 'name_2'])
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_list'), scatter_data_disc.get_data_in()['name_list']['value'])
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.name_list'), disc2.get_data_in()['name_list'])

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.y_dict'), {'name_1': 10, 'name_2': 16})
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.y_dict'), disc1.get_data_out()['y_dict'])

        pd.testing.assert_frame_equal(self.exec_eng.dm.get_value(
            'MyCase.Disc2.name_1.df'), pd.DataFrame({
                'name': ['name_1', 'name_1'],
                'c1': [1.0, 1.0],
                'c2': [5.0, 5.0],
            }))

        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.Disc2.name_1.df'), scatter_data_disc.get_data_out()['name_1.df'])

        pd.testing.assert_frame_equal(self.exec_eng.dm.get_value(
            'MyCase.Disc2.name_2.df'), pd.DataFrame({
                'name': ['name_2', 'name_2'],
                'c1': [10.0, 10.0],
                'c2': [50.0, 50.0],
            }))
        self.assertEqual(self.exec_eng.dm.get_data(
            'MyCase.Disc2.name_2.df'), scatter_data_disc.get_data_out()['name_2.df'])

        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.t_dict'), {'name_1': 4, 'name_2': 5})

        # test configure/run after name_list modification

        dict_values = {self.study_name + '.name_list': ['name_1']}
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()

        # test scatter data cleaning after configure
        self.assertNotIn('MyCase.Disc2.name_2.df',
                         self.exec_eng.dm.data_id_map.keys())

        # check referencing of scatter_data output
        disc2 = self.exec_eng.dm.get_disciplines_with_name('MyCase.Disc2.name_1')[
            0]
        scatter_data = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.scatter_data')[0]
        full_name_df = 'MyCase.Disc2.name_1.df'
        dm_referencing = self.exec_eng.dm.get_data(full_name_df)
        self.assertEqual(disc2.get_var_full_name(
            'df', disc2._data_in), full_name_df)
        self.assertEqual(scatter_data.get_var_full_name(
            'name_1.df', scatter_data._data_out), full_name_df)

        # TO DO: correct referencing of disc2 input
        # self.assertTrue(disc2._data_in['y'] is dm_referencing)
        self.assertTrue(scatter_data._data_out['name_1.df'] is dm_referencing)

    def test_05_scatter_data_with_lists_in_map_and_subtypes(self):
        ns_dict = {'ns_ac': self.namespace,
                   'ns_data_out': self.namespace}
        self.exec_eng.ns_manager.add_ns_def(ns_dict)

        mydict_build = {'input_name': 'name_list',

                        'input_ns': 'ns_barrierr',
                        'output_name': 'ac_name',
                        'scatter_ns': 'ns_ac'}

        mydict_data = {'input_name': ['y_dict', 'dict_float_dict', 'list_float_dict', 'list_dict_float_dict'],
                       'input_type': ['dict', 'dict', 'dict', 'dict'],
                       'input_ns': 'ns_ac',
                       'output_name': ['y', 'dict_float', 'list_float', 'list_dict_float'],
                       'output_ns': 'ns_data_out',
                       'output_type': ['float', 'dict', 'list', 'list'],
                       'scatter_var_name': 'name_list'}

        self.exec_eng.ns_manager.add_ns('ns_barrierr', 'MyCase')
        self.exec_eng.smaps_manager.add_build_map('name_list', mydict_build)
        self.exec_eng.smaps_manager.add_data_map('data_map', mydict_data)

        disc2_builder = self.factory.get_builder_from_module(
            'Disc2', self.mod2_scatter_gather)

        scatter_build = self.exec_eng.factory.create_scatter_builder(
            'Disc2', 'name_list', disc2_builder)

        scatter_data = self.exec_eng.factory.create_scatter_data_builder(
            'scatter_data', 'data_map')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            [scatter_build, scatter_data])

        self.exec_eng.configure()
        dict_values = {self.study_name + '.name_list': ['name_1', 'name_2'],
                       self.study_name + '.y_dict': {'name_1': 1.0, 'name_2': 2.0},
                       self.study_name + '.dict_float_dict': {'name_1': {'a': 1.0, 'b': 2.0},
                                                              'name_2': {'a': 1.0, 'b': 2.0}},
                       self.study_name + '.list_float_dict': {'name_1': [1.0, 2.0],
                                                              'name_2': [1.0, 2.0]},
                       self.study_name + '.list_dict_float_dict': {
                           'name_1': [{'a': 1.0, 'b': 2.0}, {'c': 3.0, 'd': 4.0}],
                           'name_2': [{'a': 1.0, 'b': 2.0}, {'c': 3.0, 'd': 4.0}]},
                       }
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.execute()

        # test dm.data_dict content and data_in/data_out referencing in dm

        scatter_data_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.scatter_data')[0]

        # assert that scatter data discipline has correct inputs and outputs
        self.assertListEqual(
            [key for key in list(scatter_data_disc._data_in.keys()) if key not in scatter_data_disc.NUM_DESC_IN], [
                'y_dict', 'dict_float_dict', 'list_float_dict', 'list_dict_float_dict', 'name_list', ])
        self.assertListEqual(list(scatter_data_disc._data_out.keys()),
                             ['name_1.y', 'name_1.dict_float', 'name_1.list_float', 'name_1.list_dict_float',
                              'name_2.y', 'name_2.dict_float', 'name_2.list_float', 'name_2.list_dict_float'])

        # assert that scatter data discipline has correct values to distribute
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.y_dict'), {'name_1': 1.0, 'name_2': 2.0})
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.dict_float_dict'), {'name_1': {'a': 1.0, 'b': 2.0}, 'name_2': {'a': 1.0, 'b': 2.0}})
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.list_float_dict'), {'name_1': [1.0, 2.0], 'name_2': [1.0, 2.0]})
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.list_dict_float_dict'), {'name_1': [{'a': 1.0, 'b': 2.0}, {'c': 3.0, 'd': 4.0}],
                                             'name_2': [{'a': 1.0, 'b': 2.0}, {'c': 3.0, 'd': 4.0}]})

        # assert that scatter data discipline has correct subtype descriptors
        self.assertDictEqual(self.exec_eng.dm.get_data(
            'MyCase.y_dict', 'subtype_descriptor'), {'dict': 'float'})
        self.assertDictEqual(self.exec_eng.dm.get_data(
            'MyCase.dict_float_dict', 'subtype_descriptor'), {'dict': {'dict': 'float'}})
        self.assertDictEqual(self.exec_eng.dm.get_data(
            'MyCase.list_float_dict', 'subtype_descriptor'), {'dict': {'list': 'float'}})
        self.assertDictEqual(self.exec_eng.dm.get_data(
            'MyCase.list_dict_float_dict', 'subtype_descriptor'), {'dict': {'list': {'dict': 'float'}}})

        # assert that disciplines's subtype descriptors are correct
        self.assertDictEqual(self.exec_eng.dm.get_data(
            'MyCase.name_1.dict_float', self.SUBTYPE), {'dict': 'float'})
        self.assertDictEqual(self.exec_eng.dm.get_data(
            'MyCase.name_1.list_float', self.SUBTYPE), {'list': 'float'})
        self.assertDictEqual(self.exec_eng.dm.get_data(
            'MyCase.name_1.list_dict_float', self.SUBTYPE), {'list': {'dict': 'float'}})

        # assert that scattered discipline is fed with correct inputs values by
        # scatter data
        self.assertEqual(self.exec_eng.dm.get_value(
            'MyCase.name_1.y'), 1.0)
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.name_1.dict_float'), {'a': 1.0, 'b': 2.0})
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.name_1.list_float'), [1.0, 2.0])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.name_1.list_dict_float'), [{'a': 1.0, 'b': 2.0}, {'c': 3.0, 'd': 4.0}])
        self.exec_eng.load_study_from_input_dict({self.study_name + '.name_list': [],
                                                  self.study_name + '.y_dict': {}})


if __name__ == "__main__":
    cls = TestScatterData()
    cls.setUp()
    cls.test_05_scatter_data_with_lists_in_map_and_subtypes()
