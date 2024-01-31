'''
Copyright 2022 Airbus SAS
Modifications on 2023/02/23-2023/11/03 Copyright 2023 Capgemini

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
from sostrades_core.tools.conversion.conversion_sostrades_sosgemseo import convert_array_into_df, \
    convert_array_into_new_type, convert_new_type_into_array, DEFAULT_EXCLUDED_COLUMNS
from gemseo.utils.compare_data_manager_tooling import dict_are_equal

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
import cProfile
import pstats
from io import StringIO
import pandas as pd
import numpy as np
from numpy import array
from pandas.testing import assert_frame_equal

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestExtendDataframe(unittest.TestCase):
    """
    Extend dataframe type for GEMSEO test class
    """

    def setUp(self):

        self.name = 'EE'

    def test_01_extend_df_sosdiscipline(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            disc6_builder)
        exec_eng.configure()

        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.df'] = pd.DataFrame(
            array([[0.5, 0.5]]), columns=['c1', 'c2'])
        values_dict['EE.dict_df'] = {'key_1': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2']),
                                     'key_2': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2'])}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.display_treeview_nodes()
        exec_eng.execute()

        target = {'EE.df': array([0.5, 0.5]), 'EE.dict_df': array([5., 3., 5., 3.]), 'EE.h': array([0.75, 0.75])}

        data_dm = {key: exec_eng.dm.get_value(key) for key in target.keys()}
        converted_data_dm = convert_new_type_into_array(data_dm, exec_eng.dm)

        # check new_types conversion into array
        self.assertTrue(dict_are_equal(converted_data_dm, target))

        reconverted_data_dm = convert_array_into_new_type(converted_data_dm, exec_eng.dm)

        # check array conversion into new_types
        self.assertTrue(dict_are_equal(data_dm, reconverted_data_dm))

    def test_02_extend_df_soscoupling(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()
        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.h'] = [8., 9.]
        values_dict['EE.df'] = pd.DataFrame(
            array([[5., 3.]]), columns=['c1', 'c2'])
        values_dict['EE.dict_df'] = {'key_1': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2']),
                                     'key_2': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2'])}
        exec_eng.load_study_from_input_dict(values_dict)

        target = {'EE.h': array([8., 9.]), 'EE.dict_df': array([5., 3., 5., 3.]), 'EE.df': array([5., 3.])}

        data_dm = {key: exec_eng.dm.get_value(key) for key in target.keys()}
        converted_data_dm = convert_new_type_into_array(data_dm, exec_eng.dm)

        # check new_types conversion into array
        self.assertTrue(dict_are_equal(converted_data_dm, target))

        reconverted_data_dm = convert_array_into_new_type(converted_data_dm, exec_eng.dm)

        # check array conversion into new_types
        self.assertTrue(dict_are_equal(data_dm, reconverted_data_dm))

        exec_eng.execute()

        disc7 = exec_eng.root_process.proxy_disciplines[1]

        tolerance = exec_eng.dm.get_value('EE.tolerance')

        df = disc7.get_sosdisc_outputs('df')
        df_target = pd.DataFrame(
            array([[np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0]]), columns=['c1', 'c2'])
        assert_frame_equal(df, df_target, check_exact=False, rtol=1e-5)

        max_mda_iter = exec_eng.dm.get_value('EE.max_mda_iter')
        residual_history = exec_eng.root_process.mdo_discipline_wrapp.mdo_discipline.sub_mda_list[0].residual_history

        # Check residual history
        self.assertLessEqual(len(residual_history), max_mda_iter)
        self.assertLessEqual(residual_history[-1][0], tolerance)

    def test_03_check_df_excluded_columns(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            disc6_builder)
        exec_eng.configure()

        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.df'] = pd.DataFrame(
            array([[2020, 2020, 0.5, 0.5]]), columns=['years', 'year', 'c1', 'c2'])
        values_dict['EE.dict_df'] = {'key_1': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2']),
                                     'key_2': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2'])}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.display_treeview_nodes()
        exec_eng.execute()

        # -- check dataframe as SoSTrades output
        disc6 = exec_eng.root_process.proxy_disciplines[0]

        # convert dataframe into array and check data converted and metadata
        data_dict_converted = convert_new_type_into_array({'EE.df': exec_eng.dm.get_value('EE.df')}, exec_eng.dm)
        self.assertListEqual(list(data_dict_converted['EE.df']), [0.5, 0.5])
        metadata = exec_eng.dm.get_data('EE.df', 'type_metadata')[0]
        self.assertListEqual(metadata['years'], [2020.0])
        self.assertListEqual(metadata['year'], [2020.0])
        self.assertListEqual(metadata['__columns__'].values.tolist(), ['c1', 'c2'])

        # Check that in SoSTrades we have the columns back and at the right
        # order
        df = disc6.get_sosdisc_inputs('df')
        df_target = pd.DataFrame(
            array([[2020, 2020, 0.5, 0.5]]), columns=['years', 'year', 'c1', 'c2'], index=[0.])
        self.assertTrue(df.equals(df_target),
                        f'expected and output DF are different:\n{df_target}\n VS\n{df}')

    def test_04_check_empty_df(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_test', self.name)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc4_dict_empty_df.Disc4EmptyDf'
        disc4_builder = exec_eng.factory.get_builder_from_module(
            'Disc4', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc5_dict_empty_df.Disc5EmptyDf'
        disc5_builder = exec_eng.factory.get_builder_from_module(
            'Disc5', mod_list)
        exec_eng.factory.set_builders_to_coupling_builder(
            [disc4_builder, disc5_builder])

        exec_eng.configure()

        # additional test to verify that values_in are used
        values_dict = {}
        values_dict['EE.h'] = {'dataframe': pd.DataFrame(data={'col1': array([0.5,
                                                                              0.5])})}
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.display_treeview_nodes()
        exec_eng.execute()

        self.assertTrue(exec_eng.dm.get_value('EE.Disc5.is_df_empty'))
        self.assertTrue(exec_eng.dm.get_value(
            'EE.Disc5.is_dict_empty_df_empty'))
        self.assertTrue(exec_eng.dm.get_value(
            'EE.Disc5.is_dict_empty_list_empty'))

    def test_05_multi_index_column_df(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        values_dict = {}

        col = pd.MultiIndex.from_tuples(
            [('c1', ''), ('c2', '')])
        df_multi_index_columns = pd.DataFrame(
            data=[[0.5, 1.0], [0.5, 1.0]], columns=col)

        values_dict['EE.df'] = df_multi_index_columns
        values_dict['EE.dict_df'] = {'key_1': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2']),
                                     'key_2': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2'])}
        exec_eng.load_study_from_input_dict(values_dict)

        disc6 = exec_eng.dm.get_disciplines_with_name('EE.Disc6')[0]

        self.assertTrue(disc6.get_sosdisc_inputs(
            'df').columns.equals(df_multi_index_columns.columns))
        self.assertTrue(disc6.get_sosdisc_inputs(
            'df').equals(df_multi_index_columns))

        df_in_dm = disc6.get_sosdisc_inputs('df')
        df_converted_array = convert_new_type_into_array(
            {'EE.df': df_in_dm}, exec_eng.dm)

        df_reconverted = convert_array_into_new_type(df_converted_array, exec_eng.dm)
        self.assertTrue(df_reconverted['EE.df'].equals(df_multi_index_columns))
        self.assertTrue(df_reconverted['EE.df'].columns.equals(col))
        self.assertTrue(exec_eng.dm.get_data(
            'EE.df', 'type_metadata')[0]['__columns__'].equals(col))

        exec_eng.execute()

        target = [0.7071067811865475, 0.7071067811865475]
        self.assertListEqual(list(exec_eng.dm.get_value('EE.h')), target)

    def test_06_multi_index_rows_df(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)
        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc7.Disc7'
        disc7_builder = exec_eng.factory.get_builder_from_module(
            'Disc7', mod_list)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            [disc6_builder, disc7_builder])
        exec_eng.configure()

        values_dict = {}

        mux = pd.MultiIndex.from_arrays(
            [list('aaabbbccc'), [0, 1, 2, 0, 1, 2, 0, 1, 2]], names=['one', 'two'])
        df = pd.DataFrame({'c1': [0.5] * 9, 'c2': [0.5] * 9}, mux)

        values_dict['EE.df'] = df
        values_dict['EE.dict_df'] = {'key_1': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2']),
                                     'key_2': pd.DataFrame(array([[5., 3.]]), columns=['c1', 'c2'])}
        exec_eng.load_study_from_input_dict(values_dict)

        disc6 = exec_eng.dm.get_disciplines_with_name('EE.Disc6')[0]

        self.assertTrue(disc6.get_sosdisc_inputs(
            'df').index.equals(mux))
        self.assertTrue(disc6.get_sosdisc_inputs(
            'df').equals(df))

        df_in_dm = disc6.get_sosdisc_inputs('df')
        df_converted_array = convert_new_type_into_array(
            {'EE.df': df_in_dm}, exec_eng.dm)
        df_reconverted = convert_array_into_new_type(df_converted_array, exec_eng.dm)
        self.assertTrue(df_reconverted['EE.df'].equals(df))
        self.assertTrue(df_reconverted['EE.df'].index.equals(mux))
        self.assertTrue(exec_eng.dm.get_data(
            'EE.df', 'type_metadata')[0]['__indices__'].equals(mux))

        exec_eng.execute()

        target = [0.7071067811865475, 0.7071067811865475]
        self.assertListEqual(list(exec_eng.dm.get_value('EE.h')), target)

    def test_07_convert_array_into_df(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            disc6_builder)
        exec_eng.configure()
        disc6 = exec_eng.root_process.proxy_disciplines[0]
        metadata = {'years': list(np.arange(2020, 2101)),
                    '__key__': [], '__type__': pd.DataFrame,
                    '__columns__': ['CO2', 'uranium fuel', 'biomass_dry', 'wet_biomass',
                                    'wood', 'carbon', 'crude oil', 'NaturalOil', 'Methanol',
                                    'Sodium_Hydroxyde', 'Potassium_Hydroxyde', 'oxygen',
                                    'calcium', 'potassium', 'amine', 'sea_water', 'water'],
                    '__shape__': (81, 17), '__size__': 1377,
                    '__dtypes__': [np.array([]).dtype] * 17}
        arr_to_convert = [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.4, 47.8, 49.2, 50.6, 52.0, 54.2, 56.4, 58.6, 60.8,
                          63.0, 65.2, 67.4,
                          69.6, 71.8, 74.0, 78.4, 82.8, 87.2, 91.6, 96.0, 100.6, 105.2, 109.8, 114.4, 119.0, 123.6,
                          128.2, 132.8, 137.4,
                          142.0, 146.6, 151.2, 155.8, 160.39999999999998, 165.0, 169.6, 174.2, 178.8,
                          183.39999999999998, 188.0, 192.6,
                          197.2, 201.8, 206.39999999999998, 211.0, 215.6, 220.2, 224.79999999999998, 229.39999999999998,
                          234.0, 238.6,
                          243.2, 247.79999999999998, 252.39999999999998, 257.0, 261.6, 266.2, 270.79999999999995, 275.4,
                          280.0, 284.6,
                          289.2, 293.79999999999995, 298.4, 303.0, 307.6, 312.2, 316.79999999999995, 321.4, 326.0,
                          330.6, 335.2,
                          339.79999999999995, 344.4, 349.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0,
                          1390000.0, 1390000.0,
                          1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0,
                          1390000.0, 1390000.0,
                          1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0,
                          1390000.0, 1390000.0,
                          1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0,
                          1390000.0, 1390000.0,
                          1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0,
                          1390000.0, 1390000.0,
                          1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0,
                          1390000.0, 1390000.0,
                          1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0,
                          1390000.0, 1390000.0,
                          1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0, 1390000.0,
                          1390000.0, 1390000.0,
                          1390000.0, 1390000.0, 1390000.0, 1390000.0, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12,
                          68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 68.12, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
                          56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
                          56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
                          56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
                          56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
                          56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
                          56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0,
                          56.0, 56.0, 56.0, 56.0, 56.0, 120.0, 120.0, 120.0,
                          120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
                          120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
                          120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
                          120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
                          120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
                          120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
                          120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0,
                          120.0, 120.0, 120.0, 120.0, 120.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0,
                          25000.0, 25000.0, 25000.0, 25000.0, 25000.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0,
                          44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0,
                          44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0,
                          44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0,
                          44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0,
                          44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0,
                          44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 44.0,
                          44.0, 44.0, 44.0, 44.0, 44.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0,
                          1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 1100.0, 400.0,
                          400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                          400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                          400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                          400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                          400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                          400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
                          400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 425.0, 425.0, 425.0, 425.0,
                          425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0,
                          425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0,
                          425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0,
                          425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0,
                          425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0,
                          425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 425.0, 772.0,
                          772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0,
                          772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0,
                          772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0,
                          772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0,
                          772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0,
                          772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0, 772.0,
                          772.0, 772.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                          60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                          60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                          60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                          60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                          60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0,
                          85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0,
                          85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0,
                          85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0,
                          85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0,
                          85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 500.0,
                          500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
                          500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
                          500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
                          500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
                          500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
                          500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
                          500.0, 500.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0,
                          1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0,
                          1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0,
                          1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0,
                          1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0,
                          1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0,
                          1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1300.0,
                          1300.0, 1300.0, 1300.0, 1300.0, 1300.0, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313,
                          1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313,
                          1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313,
                          1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313,
                          1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313,
                          1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313,
                          1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313,
                          1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.4313, 1.78, 1.78, 1.78,
                          1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78,
                          1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78,
                          1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78,
                          1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78,
                          1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78, 1.78,
                          1.78, 1.78, 1.78]

        df = convert_array_into_df(
            np.array(arr_to_convert), metadata)
        exec_eng.load_study_from_input_dict({'EE.dict_df': df})
        old_value = exec_eng.dm.get_value('EE.dict_df')

        profil = cProfile.Profile()
        profil.enable()

        for i in range(1000):
            convert_array_into_df(
                np.array(arr_to_convert), metadata)

        profil.disable()
        result = StringIO()

        ps = pstats.Stats(profil, stream=result)
        ps.sort_stats('cumulative')
        ps.print_stats(100)
        result = result.getvalue()
        # chop the string into a csv-like buffer
        result = 'ncalls' + result.split('ncalls')[-1]
        result = '\n'.join([','.join(line.rstrip().split(None, 5))
                            for line in result.split('\n')])

        print(result)

    def test_08_convert_array_into_small_df(self):

        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            disc6_builder)
        exec_eng.configure()
        disc6 = exec_eng.root_process.proxy_disciplines[0]
        metadata = {'years': list(np.arange(2020, 2101)),
                    '__key__': [], '__type__': pd.DataFrame,
                    '__columns__': ['CO2'],
                    '__shape__': (81, 1), '__size__': 81,
                    '__dtypes__': [np.array([]).dtype]}
        arr_to_convert = [40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.4, 47.8, 49.2, 50.6, 52.0, 54.2, 56.4, 58.6, 60.8,
                          63.0, 65.2, 67.4,
                          69.6, 71.8, 74.0, 78.4, 82.8, 87.2, 91.6, 96.0, 100.6, 105.2, 109.8, 114.4, 119.0, 123.6,
                          128.2, 132.8, 137.4,
                          142.0, 146.6, 151.2, 155.8, 160.39999999999998, 165.0, 169.6, 174.2, 178.8,
                          183.39999999999998, 188.0, 192.6,
                          197.2, 201.8, 206.39999999999998, 211.0, 215.6, 220.2, 224.79999999999998, 229.39999999999998,
                          234.0, 238.6,
                          243.2, 247.79999999999998, 252.39999999999998, 257.0, 261.6, 266.2, 270.79999999999995, 275.4,
                          280.0, 284.6,
                          289.2, 293.79999999999995, 298.4, 303.0, 307.6, 312.2, 316.79999999999995, 321.4, 326.0,
                          330.6, 335.2,
                          339.79999999999995, 344.4, 349.0]
        df = convert_array_into_df(
            np.array(arr_to_convert), metadata)
        exec_eng.load_study_from_input_dict({'EE.dict_df': df})
        old_value = exec_eng.dm.get_value('EE.dict_df')

        profil = cProfile.Profile()
        profil.enable()

        for i in range(1000):
            convert_array_into_df(
                np.array(arr_to_convert), metadata)

        profil.disable()
        result = StringIO()

        ps = pstats.Stats(profil, stream=result)
        ps.sort_stats('cumulative')
        ps.print_stats(100)
        result = result.getvalue()
        # chop the string into a csv-like buffer
        result = 'ncalls' + result.split('ncalls')[-1]
        result = '\n'.join([','.join(line.rstrip().split(None, 5))
                            for line in result.split('\n')])

        print(result)

        df = exec_eng.dm.get_value('EE.dict_df')
        self.assertListEqual(df['CO2'].values.tolist(),
                             arr_to_convert)
        self.assertListEqual(df['years'].values.tolist(),
                             metadata['years'])

    def test_09_df_dtype_filtering(self):
        exec_eng = ExecutionEngine(self.name)

        exec_eng.ns_manager.add_ns('ns_protected', self.name)

        mod_list = 'sostrades_core.sos_wrapping.test_discs.disc6.Disc6'
        disc6_builder = exec_eng.factory.get_builder_from_module(
            'Disc6', mod_list)

        exec_eng.factory.set_builders_to_coupling_builder(
            disc6_builder)
        exec_eng.configure()
        disc6 = exec_eng.root_process.proxy_disciplines[0]
        metadata = {'__type__': pd.DataFrame,
                    '__columns__': ['name', 'age', 'weight', 'adult', 'favorite complex number', 'satisfaction'],
                    '__shape__': (5, 6), '__size__': 30,
                    '__dtypes__': [str, int, float, bool, complex, float]}
        arr_to_convert = [['Michel', 30, 50.3, 'True', 1.2j, -1.0],
                          ['Anne', 55, 62.4, 'True', 47j, 8],
                          ['Farid', 23, 81.9, 'True', 0.3j, -6.0],
                          ['Yu', 10, 30.6, 'False', 4.7j, 3],
                          ['Erling', 17, 62.3, 'False', 69j, 10]]
        df_init = convert_array_into_df(
            np.array(arr_to_convert), metadata)
        exec_eng.load_study_from_input_dict({'EE.df': df_init})
        old_value = exec_eng.dm.get_value('EE.df')

        profil = cProfile.Profile()
        profil.enable()

        for i in range(1000):
            convert_array_into_df(
                np.array(arr_to_convert), metadata)

        profil.disable()
        result = StringIO()

        ps = pstats.Stats(profil, stream=result)
        ps.sort_stats('cumulative')
        ps.print_stats(100)
        result = result.getvalue()
        # chop the string into a csv-like buffer
        result = 'ncalls' + result.split('ncalls')[-1]
        result = '\n'.join([','.join(line.rstrip().split(None, 5))
                            for line in result.split('\n')])

        print(result)

        df_after_exec = exec_eng.dm.get_value('EE.df')
        self.assertDictEqual(df_after_exec.to_dict(), df_init.to_dict())
        self.assertListEqual(
            list(df_after_exec.dtypes),
            [object, int, float, bool, complex, float]
        )
        converted_array = convert_new_type_into_array({'EE.df': df_after_exec}, exec_eng.dm.reduced_dm)['EE.df']
        self.assertEqual(converted_array.size, (6 - 2) * 5)

        excluded_col = exec_eng.dm.reduced_dm['EE.df']['dataframe_excluded_columns']
        self.assertListEqual(excluded_col, DEFAULT_EXCLUDED_COLUMNS + ['name', 'adult'])


if '__main__' == __name__:
    cls = TestExtendDataframe()
    cls.setUp()
    cls.test_08_convert_array_into_small_df()
