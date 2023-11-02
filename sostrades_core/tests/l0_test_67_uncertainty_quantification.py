'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2023/11/02 Copyright 2023 Capgemini

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
from sostrades_core.study_manager.base_study_manager import BaseStudyManager

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
'''
import unittest
from os.path import join, dirname
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd

from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class TestUncertaintyQuantification(unittest.TestCase):
    """
    UncertaintyQuantification test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.repo = 'sostrades_core.sos_processes.test'
        self.mod_path = 'sostrades_core.sos_wrapping.analysis_discs.uncertainty_quantification'
        self.proc_name = 'test_uncertainty_quantification_analysis'
        self.name = 'Test'
        self.uncertainty_quantification = 'UncertaintyQuantification'

        self.ee = ExecutionEngine(self.name)
        self.factory = self.ee.factory
        self.dir_to_del = []

    def tearDown(self):
        for dir in self.dir_to_del:
            if Path(dir).is_dir():
                rmtree(dir)

    def test_01_uncertainty_quantification(self):

        builder = self.factory.get_builder_from_process(
            self.repo, self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {'ns_eval': f'{self.name}.{self.uncertainty_quantification}',
                   'ns_uncertainty_quantification': f'{self.name}.UncertaintyQuantification'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.configure()

        self.ee.display_treeview_nodes()

        self.data_dir = join(dirname(dirname(
            __file__)), 'sos_processes', 'test', self.proc_name, 'data')

        self.samples_dataframe = pd.read_csv(
            join(self.data_dir, 'samples_df.csv'))

        # fixes a particular state of the random generator algorithm thanks to
        # the seed sample_size
        np.random.seed(42)

        Var1 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))
        Var2 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))

        out1 = list(pd.Series(Var1 + Var2) * 100000)
        out2 = list(pd.Series(Var1 * Var2) * 100000)
        out3 = list(pd.Series(np.square(Var1) + np.square(Var2)) * 100000)

        self.data_df = pd.DataFrame(
            {'scenario': self.samples_dataframe['scenario'], 'output1': out1, 'output2': out2, 'output3': out3})

        input_selection = {'selected_input': [True, True, True],
                           'full_name': ['COC', 'RC', 'NRC']}

        output_selection = {'selected_output': [True, True, True],
                            'full_name': ['output1', 'output2', 'output3']}

        dspace = pd.DataFrame({
            'shortest_name': ['COC', 'RC', 'NRC'],
            'lower_bnd': [85., 80., 80.],
            'upper_bnd': [105., 120., 120.],
            'nb_points': [10, 10, 10],
            'full_name': ['COC', 'RC', 'NRC'],
        })

        private_values = {
            f'{self.name}.{self.uncertainty_quantification}.samples_inputs_df': self.samples_dataframe,
            f'{self.name}.{self.uncertainty_quantification}.samples_outputs_df': self.data_df,
            f'{self.name}.{self.uncertainty_quantification}.design_space': dspace,
            f'{self.name}.{self.uncertainty_quantification}.eval_inputs': pd.DataFrame(input_selection),
            f'{self.name}.{self.uncertainty_quantification}.eval_outputs': pd.DataFrame(output_selection),
        }

        self.ee.load_study_from_input_dict(private_values)
        self.ee.configure()
        self.ee.execute()

        uncertainty_quanti_disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.uncertainty_quantification}')[0]

        uncertainty_quanti_disc_output = uncertainty_quanti_disc.get_sosdisc_outputs()
        out_df = uncertainty_quanti_disc_output['output_interpolated_values_df']

        filter = uncertainty_quanti_disc.get_chart_filter_list()
        graph_list = uncertainty_quanti_disc.get_post_processing_list(filter)
        #for graph in graph_list:
        #    graph.to_plotly().show()

    def test_02_uncertainty_quantification_from_cartesian_product(self):
        """In this test we prove the ability to couple a grid search and an uq
        """
        proc_name = 'test_coupling_doe_uq'

        builder = self.factory.get_builder_from_process(
            self.repo, proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.load_study_from_input_dict({})

        disc1_name = 'Disc1'
        ns = f'{self.ee.study_name}'
        dspace_dict = {'variable': [f'subprocess.{disc1_name}.a', 'x'],

                       'lower_bnd': [0., 0.],
                       'upper_bnd': [10., 10.],

                       }
        dspace = pd.DataFrame(dspace_dict)

        output_selection_obj_y1_y2 = {'selected_output': [True, True, False],
                                      'full_name': [f'subprocess.{disc1_name}.indicator', 'z', 'y']}
        output_selection_obj_y1_y2 = pd.DataFrame(output_selection_obj_y1_y2)

        disc_dict = {}
        # DoE inputs
        disc_dict[f'{ns}.Eval.builder_mode'] = 'mono_instance'
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = 'cartesian_product'

        a_list = np.linspace(0, 10, 2).tolist()
        x_list = np.linspace(0, 10, 2).tolist()
        eval_inputs = pd.DataFrame({'selected_input': [True, False, True, False],
                                    'full_name': [f'subprocess.{disc1_name}.a',
                                                  f'subprocess.{disc1_name}.b',
                                                  f'x',
                                                  f'subprocess.Disc2.power']})

        eval_inputs_cp = pd.DataFrame({'selected_input': [True, False, True, False],
                                       'full_name': [f'subprocess.{disc1_name}.a',
                                                     f'subprocess.{disc1_name}.b',
                                                     f'x',
                                                     f'subprocess.Disc2.power'],
                                       'list_of_values': [a_list, [], x_list, []]
                                       })
        disc_dict[f'{self.ee.study_name}.Eval.eval_inputs_cp'] = eval_inputs_cp
        disc_dict[f'{self.ee.study_name}.Eval.eval_inputs'] = eval_inputs
        disc_dict[f'{ns}.Eval.design_space'] = dspace
        # disc_dict[f'{ns}.Eval.eval_inputs'] = input_selection_x_z
        disc_dict[f'{ns}.Eval.eval_outputs'] = output_selection_obj_y1_y2

        disc_dict[f'{ns}.Eval.x'] = 10.
        disc_dict[f'{ns}.Eval.subprocess.{disc1_name}.a'] = 5.
        disc_dict[f'{ns}.Eval.subprocess.{disc1_name}.b'] = 2.
        disc_dict[f'{ns}.Eval.subprocess.Disc2.constant'] = 3.1416
        disc_dict[f'{ns}.Eval.subprocess.Disc2.power'] = 2

        self.ee.load_study_from_input_dict(disc_dict)

        self.ee.execute()

    def test_03_uncertainty_quantification_with_arrays_in_inputs(self):
        """This tests evaluates the capacity to perform uncertainty quantification when some inputs are arrays"""
        builder = self.factory.get_builder_from_process(
            self.repo, self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {'ns_eval': f'{self.name}.{self.uncertainty_quantification}',
                   'ns_uncertainty_quantification': f'{self.name}.UncertaintyQuantification'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.configure()

        self.ee.display_treeview_nodes()

        self.data_dir = join(dirname(dirname(
            __file__)), 'sos_processes', 'test', self.proc_name, 'data')

        self.samples_dataframe = pd.read_csv(
            join(self.data_dir, 'samples_df.csv'))

        # builds a sample dataframe with regular arrays samples
        x_range = np.arange(50, 150, 15)
        y_range = np.arange(40, 80, 8)
        z_range = np.arange(60, 120, 15)
        x, y, z = np.meshgrid(x_range, y_range, z_range)
        triplets = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        samples_dataframe = pd.concat([self.samples_dataframe[:-1]] * len(triplets))
        samples_dataframe = pd.concat([samples_dataframe, self.samples_dataframe.iloc[-1:]])
        array_var_column = []
        for triplet in triplets:
            array_var_column += [triplet] * len(self.samples_dataframe[:-1])
        array_var_column.append(10.)
        samples_dataframe['input_array'] = array_var_column
        samples_dataframe['scenario'] = [f'scenario_{i}' for i in range(len(samples_dataframe) - 1)] + ['reference']
        self.samples_dataframe = samples_dataframe

        # fixes a particular state of the random generator algorithm thanks to
        # the seed sample_size
        np.random.seed(42)

        Var1 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))
        Var2 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))

        out1 = list(pd.Series(Var1 + Var2) * 100000)
        out2 = list(pd.Series(Var1 * Var2) * 100000)
        out3 = list(pd.Series(np.square(Var1) + np.square(Var2)) * 100000)

        self.data_df = pd.DataFrame(
            {'scenario': self.samples_dataframe['scenario'], 'output1': out1, 'output2': out2, 'output3': out3})

        input_selection = {'selected_input': [True, True, True, True],
                           'full_name': ['COC', 'RC', 'NRC', 'input_array']}

        output_selection = {'selected_output': [True, True, True],
                            'full_name': ['output1', 'output2', 'output3']}

        dspace = pd.DataFrame({
            'shortest_name': ['COC', 'RC', 'NRC', 'input_array'],
            'lower_bnd': [85., 80., 80., np.array([50., 40., 60])],
            'upper_bnd': [105., 120., 120., np.array([150., 80., 120.])],
            'nb_points': [10, 10, 10, 10],
            'full_name': ['COC', 'RC', 'NRC', 'input_array'],
        })

        private_values = {
            f'{self.name}.{self.uncertainty_quantification}.samples_inputs_df': self.samples_dataframe,
            f'{self.name}.{self.uncertainty_quantification}.samples_outputs_df': self.data_df,
            f'{self.name}.{self.uncertainty_quantification}.design_space': dspace,
            f'{self.name}.{self.uncertainty_quantification}.eval_inputs': pd.DataFrame(input_selection),
            f'{self.name}.{self.uncertainty_quantification}.eval_outputs': pd.DataFrame(output_selection),
        }

        self.ee.load_study_from_input_dict(private_values)
        self.ee.configure()
        self.ee.execute()

        uncertainty_quanti_disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.uncertainty_quantification}')[0]

        uncertainty_quanti_disc_output = uncertainty_quanti_disc.get_sosdisc_outputs()
        out_df = uncertainty_quanti_disc_output['output_interpolated_values_df']

        filter = uncertainty_quanti_disc.get_chart_filter_list()
        graph_list = uncertainty_quanti_disc.get_post_processing_list(filter)
        """
        for graph in graph_list:
            graph.to_plotly().show()
        """

    def test_04_uncertainty_quantification_with_arrays_in_input_and_outputs(self):
        """This tests evaluates the capacity to perform uncertainty quantification when some inputs are arrays and
        some outputs are arrays"""
        builder = self.factory.get_builder_from_process(
            self.repo, self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {'ns_eval': f'{self.name}.{self.uncertainty_quantification}',
                   'ns_uncertainty_quantification': f'{self.name}.UncertaintyQuantification'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.configure()

        self.ee.display_treeview_nodes()

        self.data_dir = join(dirname(dirname(
            __file__)), 'sos_processes', 'test', self.proc_name, 'data')

        self.samples_dataframe = pd.read_csv(
            join(self.data_dir, 'samples_df.csv'))

        # builds a sample dataframe with regular inputs arrays
        x_range = np.arange(50, 150, 15)
        y_range = np.arange(40, 80, 8)
        z_range = np.arange(60, 120, 15)
        x, y, z = np.meshgrid(x_range, y_range, z_range)
        triplets = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        samples_dataframe = pd.concat([self.samples_dataframe[:-1]] * len(triplets))
        samples_dataframe = pd.concat([samples_dataframe, self.samples_dataframe.iloc[-1:]])
        array_var_column = []
        for triplet in triplets:
            array_var_column += [triplet] * len(self.samples_dataframe[:-1])
        array_var_column.append(10.)
        samples_dataframe['input_array'] = array_var_column
        samples_dataframe['scenario'] = [f'scenario_{i}' for i in range(len(samples_dataframe) - 1)] + ['reference']
        self.samples_dataframe = samples_dataframe

        # fixes a particular state of the random generator algorithm thanks to
        # the seed sample_size
        np.random.seed(42)

        Var1 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))
        Var2 = np.random.uniform(-1, 1, size=len(self.samples_dataframe))

        # set outputs to be both floats and arrays
        out1 = list(pd.Series(Var1 + Var2) * 100000)
        out_array = list(np.array([(Var1*Var2)*100_000,
                                   (Var1**2 + Var2**2) * 100_000,
                                   (Var1**4 - Var2**2) * 100_000,
                                   (-Var1**2 - Var2**2) * 100_000]).T)

        self.data_df = pd.DataFrame(
            {'scenario': self.samples_dataframe['scenario'], 'output1': out1, 'output_array': out_array})

        input_selection = {'selected_input': [True, True, True, True],
                           'full_name': ['COC', 'RC', 'NRC', 'input_array']}

        output_selection = {'selected_output': [True, True],
                            'full_name': ['output1', 'output_array']}

        dspace = pd.DataFrame({
            'shortest_name': ['COC', 'RC', 'NRC', 'input_array'],
            'lower_bnd': [85., 80., 80., np.array([50., 40., 60])],
            'upper_bnd': [105., 120., 120., np.array([150., 80., 120.])],
            'nb_points': [10, 10, 10, 10],
            'full_name': ['COC', 'RC', 'NRC', 'input_array'],
        })

        private_values = {
            f'{self.name}.{self.uncertainty_quantification}.samples_inputs_df': self.samples_dataframe,
            f'{self.name}.{self.uncertainty_quantification}.samples_outputs_df': self.data_df,
            f'{self.name}.{self.uncertainty_quantification}.design_space': dspace,
            f'{self.name}.{self.uncertainty_quantification}.eval_inputs': pd.DataFrame(input_selection),
            f'{self.name}.{self.uncertainty_quantification}.eval_outputs': pd.DataFrame(output_selection),
        }

        self.ee.load_study_from_input_dict(private_values)
        self.ee.configure()
        self.ee.execute()

        uncertainty_quanti_disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.uncertainty_quantification}')[0]

        uncertainty_quanti_disc_output = uncertainty_quanti_disc.get_sosdisc_outputs()
        out_df = uncertainty_quanti_disc_output['output_interpolated_values_df']

        filter = uncertainty_quanti_disc.get_chart_filter_list()
        graph_list = uncertainty_quanti_disc.get_post_processing_list(filter)
        """
        for graph in graph_list:
            graph.to_plotly().show()
        """


    #
    # def test_03_simple_cache_on_grid_search_uq_process(self):
    #     """In this test we prove the ability of the cache to work properly on a grid search
    #     First, we create a process made of a coupling of a grid search and an uq on the grid search's output
    #     Then we activate the cache , change one uq input while maintaining grid search inputs  and run the process.
    #     since none of the grid search's inputs has been changed, we expect the grid search not to run
    #     """
    #     proc_name = 'test_coupling_doe_uq'
    #
    #     builder = self.factory.get_builder_from_process(
    #         self.repo, proc_name)
    #
    #     self.ee.factory.set_builders_to_coupling_builder(builder)
    #
    #     self.ee.load_study_from_input_dict({})
    #
    #     print(self.ee.display_treeview_nodes())
    #
    #     self.grid_search = 'GridSearch'
    #     self.study_name = 'Test'
    #
    #     eval_inputs = self.ee.dm.get_value(
    #         f'{self.study_name}.{self.grid_search}.eval_inputs')
    #     eval_inputs.loc[eval_inputs['full_name'] ==
    #                     f'{self.grid_search}.Disc1.x', ['selected_input']] = True
    #     eval_inputs.loc[eval_inputs['full_name'] ==
    #                     f'{self.grid_search}.Disc1.j', ['selected_input']] = True
    #
    #     eval_outputs = self.ee.dm.get_value(
    #         f'{self.study_name}.{self.grid_search}.eval_outputs')
    #     eval_outputs.loc[eval_outputs['full_name'] ==
    #                      f'{self.grid_search}.Disc1.y', ['selected_output']] = True
    #
    #     dspace = pd.DataFrame({
    #         'shortest_name': ['x', 'j'],
    #         'lower_bnd': [5., 20.],
    #         'upper_bnd': [7., 25.],
    #         'nb_points': [2, 2],
    #         'full_name': ['GridSearch.Disc1.x', 'GridSearch.Disc1.j'],
    #     })
    #
    #     dict_values = {
    #         # GRID SEARCH INPUTS
    #         f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
    #         f'{self.study_name}.{self.grid_search}.eval_outputs': eval_outputs,
    #         f'{self.study_name}.{self.grid_search}.design_space': dspace,
    #
    #         # DISC1 INPUTS
    #         f'{self.study_name}.{self.grid_search}.Disc1.name': 'A1',
    #         f'{self.study_name}.{self.grid_search}.Disc1.a': 20,
    #         f'{self.study_name}.{self.grid_search}.Disc1.b': 2,
    #         f'{self.study_name}.{self.grid_search}.Disc1.x': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.d': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.f': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.g': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.h': 3.,
    #         f'{self.study_name}.{self.grid_search}.Disc1.j': 3.,
    #
    #     }
    #
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     grid_search_disc = self.ee.dm.get_disciplines_with_name(
    #         f'{self.study_name}.{self.grid_search}')[0]
    #
    #     uq_disc = self.ee.dm.get_disciplines_with_name(
    #         f'{self.name}.{self.uncertainty_quantification}')[0]
    #
    #     # check cache is None
    #     self.assertEqual(
    #         grid_search_disc.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(uq_disc.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(self.ee.root_process.cache, None)
    #     self.assertEqual(self.ee.root_process.mdo_chain.cache, None)
    #     self.assertEqual(self.ee.root_process.sos_disciplines[0].cache, None)
    #
    #     # first execute
    #     res_1 = self.ee.execute()
    #     # get number of calls after first call
    #     n_call_grid_search_1 = grid_search_disc.n_calls
    #     n_call_uq_1 = uq_disc.n_calls
    #
    #     # second execute without change of parameters
    #     res_2 = self.ee.execute()
    #
    #     # get number of calls after second call
    #     n_call_grid_search_2 = grid_search_disc.n_calls
    #     n_call_uq_2 = uq_disc.n_calls
    #
    #     # self.assertEqual(n_call_root_2, n_call_root_1 + 1)
    #     self.assertEqual(n_call_grid_search_2, n_call_grid_search_1 + 1)
    #     self.assertEqual(n_call_uq_2, n_call_uq_1 + 1)
    #
    #     # ACTIVATE SIMPLE CACHE ROOT PROCESS
    #
    #     dict_values[f'{self.name}.cache_type'] = 'SimpleCache'
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     self.assertEqual(grid_search_disc.get_sosdisc_inputs(
    #         'cache_type'), 'SimpleCache')
    #     self.assertEqual(uq_disc.get_sosdisc_inputs(
    #         'cache_type'), 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.cache.__class__.__name__, 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.mdo_chain.cache.__class__.__name__, 'SimpleCache')
    #     self.assertEqual(
    #         self.ee.root_process.sos_disciplines[0].cache.__class__.__name__, 'SimpleCache')
    #
    #     # first execute
    #     res_1 = self.ee.execute()
    #     # get number of calls after first call
    #     n_call_grid_search_1 = grid_search_disc.n_calls
    #     n_call_uq_1 = uq_disc.n_calls
    #
    #     # second execute without change of parameters
    #     res_2 = self.ee.execute()
    #
    #     # get number of calls after second call
    #     n_call_grid_search_2 = grid_search_disc.n_calls
    #     n_call_uq_2 = uq_disc.n_calls
    #
    #     # check that neither grid_search nor uq has run
    #     self.assertEqual(n_call_grid_search_2, n_call_grid_search_1)
    #     self.assertEqual(n_call_uq_2, n_call_uq_1)
    #
    #     # Third execute with a change of a uq parameter and no change on doe
    #     dict_values[f'{self.study_name}.{self.grid_search}.confidence_interval'] = 95
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     res_3 = self.ee.execute()
    #
    #     # get number of calls after third call
    #     n_call_grid_search_3 = grid_search_disc.n_calls
    #     n_call_uq_3 = uq_disc.n_calls
    #
    #     # check that uq has run but grid search hasn't
    #     self.assertEqual(n_call_grid_search_3, n_call_grid_search_2)
    #     self.assertEqual(n_call_uq_3, n_call_uq_2 + 1)
    #
    #     # Fourth execute with a change of a grid_search parameter and no change
    #     # on uq
    #     dict_values[f'{self.study_name}.{self.grid_search}.wait_time_between_fork'] = 5.0
    #     self.ee.load_study_from_input_dict(dict_values)
    #     res_4 = self.ee.execute()
    #
    #     # get number of calls after fourth call
    #     n_call_grid_search_4 = grid_search_disc.n_calls
    #     n_call_uq_4 = uq_disc.n_calls
    #
    #     # check that grid search has run and uq hasn't
    #     self.assertEqual(n_call_grid_search_4, n_call_grid_search_3 + 1)
    #     self.assertEqual(n_call_uq_4, n_call_uq_3)
    #
    #     # Fifth execute with a change of a common input
    #     eval_outputs_2 = self.ee.dm.get_value('Test.GridSearch.eval_outputs')
    #     eval_outputs_2.loc[eval_outputs['full_name'] ==
    #                        f'{self.grid_search}.Disc1.indicator', ['selected_output']] = True
    #
    #     dict_values[f'{self.study_name}.{self.grid_search}.eval_outputs'] = eval_outputs_2
    #     self.ee.load_study_from_input_dict(dict_values)
    #     res_5 = self.ee.execute()
    #
    #     # get number of calls after fifth call
    #     n_call_grid_search_5 = grid_search_disc.n_calls
    #     n_call_uq_5 = uq_disc.n_calls
    #
    #     # check that both grid search and uq have run
    #     self.assertEqual(n_call_grid_search_5, n_call_grid_search_4 + 1)
    #     self.assertEqual(n_call_uq_5, n_call_uq_4 + 1)
    #
    #     # DESACTIVATE CACHE
    #
    #     dict_values[f'{self.name}.cache_type'] = 'None'
    #     self.ee.load_study_from_input_dict(dict_values)
    #
    #     # check cache is None
    #     self.assertEqual(
    #         grid_search_disc.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(uq_disc.get_sosdisc_inputs('cache_type'), 'None')
    #     self.assertEqual(self.ee.root_process.cache, None)
    #     self.assertEqual(self.ee.root_process.mdo_chain.cache, None)
    #     self.assertEqual(self.ee.root_process.sos_disciplines[0].cache, None)
    #
    #     #  execute one more time
    #     res_6 = self.ee.execute()
    #
    #     # get number of calls after changing cache
    #     n_call_grid_search_6 = grid_search_disc.n_calls
    #     n_call_uq_6 = uq_disc.n_calls
    #
    #     # check that both grid search and uq have run since there is no more
    #     # cache
    #     self.assertEqual(n_call_grid_search_6, n_call_grid_search_5 + 1)
    #     self.assertEqual(n_call_uq_6, n_call_uq_5 + 1)

    # def test_04_simple_cache_on_grid_search_uq_process(self):
    #
    #     self.ref_dir = join(dirname(__file__), 'data')
    #     self.dump_dir = join(self.ref_dir, 'dumped_cache_test_67')
    #     study_1 = study_grid_search_uq(run_usecase=True)
    #     study_1.set_dump_directory(
    #         self.dump_dir)
    #     study_1.load_data()
    #
    #     dict_values = {
    #         f'{study_1.study_name}.cache_type': 'SimpleCache'}
    #     study_1.load_data(from_input_dict=dict_values)
    #
    #     study_1.run()
    #     study_1.dump_data(self.dump_dir)
    #     study_1.dump_cache(self.dump_dir)
    #
    #     study_2 = BaseStudyManager(self.repo, 'test_coupling_doe_uq', study_1.study_name)
    #     study_2.load_data(from_path=self.dump_dir)
    #     study_2.load_cache(self.dump_dir)
    #
    #     dict_values = {
    #         f'{study_1.study_name}.{study_1.grid_search}.confidence_interval': 95}
    #     study_2.load_data(from_input_dict=dict_values)
    #
    #     study_2.run()
    #
    #     # check cache activation by counting n_calls
    #     for disc in list(study_2.ee.dm.gemseo_disciplines_id_map.values()):
    #         if disc[0].name in ['usecase', 'UncertaintyQuantification']:
    #             self.assertEqual(disc[0].n_calls, 1)
    #         else:
    #             self.assertEqual(disc[0].n_calls, 0)
    #
    #     self.dir_to_del.append(self.dump_dir)


if '__main__' == __name__:
    cls = TestUncertaintyQuantification()
    cls.setUp()
    cls.test_01_uncertainty_quantification()
    cls.tearDown()
