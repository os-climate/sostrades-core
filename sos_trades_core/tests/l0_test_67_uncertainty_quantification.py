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
from os.path import join, dirname

import numpy as np
import pandas as pd

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine


class TestUncertaintyQuantification(unittest.TestCase):
    """
    UncertaintyQuantification test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.repo = 'sos_trades_core.sos_processes.test'
        self.mod_path = 'sos_trades_core.sos_wrapping.analysis_discs.uncertainty_quantification'
        self.proc_name = 'test_uncertainty_quantification_analysis'
        self.name = 'Test'
        self.uncertainty_quantification = 'UncertaintyQuantification'

        self.ee = ExecutionEngine(self.name)
        self.factory = self.ee.factory

    def test_01_uncertainty_quantification(self):
        repo = 'sos_trades_core.sos_processes.test'
        mod_path = 'sos_trades_core.sos_wrapping.analysis_discs.uncertainty_quantification'

        builder = self.factory.get_builder_from_process(
            self.repo, self.proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {'ns_ac': f'{self.name}', 'ns_public': f'{self.name}',
                   'ns_energy': f'{self.name}',
                   'ns_coc_ac': f'{self.name}',
                   'ns_dmc_ac': f'{self.name}',
                   'ns_data_ac': f'{self.name}',
                   'ns_coc': f'{self.name}',
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

        private_values = {
            f'{self.name}.{self.uncertainty_quantification}.samples_inputs_df': self.samples_dataframe,
            f'{self.name}.{self.uncertainty_quantification}.samples_outputs_df': self.data_df,
            # f'{self.name}.{self.uncertainty_quantification}.input_distribution_parameters_df': self.input_distribution_parameters_df,
            # f'{self.name}.{self.uncertainty_quantification}.data_details_df':
            # self.data_details_df,
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
        # for graph in graph_list:
        #     graph.to_plotly().show()

    def test_02_uncertainty_quantification_from_grid_search(self):
        """In this test we prove the ability of the cache to work properly on a grid search
        First, we create a process made of a coupling of a grid search and an uq on the grid search's output
        Then we activate the cache , change one uq input while maintaining grid search inputs  and run the process.
        since none of the grid search's inputs has been changed, we expect the grid search not to run
        """
        proc_name = 'test_coupling_doe_uq'

        builder = self.factory.get_builder_from_process(
            self.repo, proc_name)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        ns_dict = {'ns_uncertainty_quantification': 'Test.GridSearch'}

        self.ee.ns_manager.add_ns_def(ns_dict)

        self.ee.load_study_from_input_dict({})

        print(self.ee.display_treeview_nodes())

        grid_search_io_path = 'Test.GridSearch'
        self.grid_search = 'GridSearch'
        self.study_name = 'Test'

        eval_inputs = self.ee.dm.get_value(f'{grid_search_io_path}.eval_inputs')
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.x', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.j', ['selected_input']] = True

        eval_outputs = self.ee.dm.get_value(
            f'{self.study_name}.{self.grid_search}.eval_outputs')
        eval_outputs.loc[eval_outputs['full_name'] ==
                         f'{self.grid_search}.Disc1.y', ['selected_output']] = True

        dspace = pd.DataFrame({
            'shortest_name': ['x', 'j'],
            'lower_bnd': [5., 20.],
            'upper_bnd': [7., 25.],
            'nb_points': [3, 3],
            'full_name': ['GridSearch.Disc1.x', 'GridSearch.Disc1.j'],
        })

        samples_inputs_df = pd.DataFrame({'scenario': {0: 'scenario_1', 1: 'scenario_2', 2: 'scenario_3',
                                                       3: 'scenario_4', 4: 'scenario_5', 5: 'scenario_6',
                                                       6: 'scenario_7', 7: 'scenario_8', 8: 'scenario_9',
                                                       9: 'reference'},
                                          'GridSearch.Disc1.x': {0: 5.0, 1: 6.0, 2: 7.0, 3: 5.0, 4: 6.0, 5: 7.0, 6: 5.0,
                                                                 7: 6.0, 8: 7.0, 9: 3.0},
                                          'GridSearch.Disc1.j': {0: 20.0, 1: 20.0, 2: 20.0, 3: 22.5, 4: 22.5, 5: 22.5,
                                                                 6: 25.0, 7: 25.0, 8: 25.0, 9: 3.0}}
                                         )
        samples_outputs_df = pd.DataFrame(
            {'y': {0: 102.0, 1: 122.0, 2: 142.0, 3: 102.0, 4: 122.0, 5: 142.0, 6: 102.0, 7: 122.0, 8: 142.0, 9: 62.0},
             'scenario': {0: 'scenario_1', 1: 'scenario_2', 2: 'scenario_3', 3: 'scenario_4', 4: 'scenario_5',
                          5: 'scenario_6', 6: 'scenario_7', 7: 'scenario_8', 8: 'scenario_9', 9: 'reference'}})

        dict_values = {
            # GRID SEARCH INPUTS
            f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
            f'{self.study_name}.{self.grid_search}.eval_outputs': eval_outputs,
            f'{self.study_name}.{self.grid_search}.design_space': dspace,

            # DISC1 INPUTS
            f'{self.study_name}.{self.grid_search}.Disc1.name': 'A1',
            f'{self.study_name}.{self.grid_search}.Disc1.a': 20,
            f'{self.study_name}.{self.grid_search}.Disc1.b': 2,
            f'{self.study_name}.{self.grid_search}.Disc1.x': 3.,
            f'{self.study_name}.{self.grid_search}.Disc1.d': 3.,
            f'{self.study_name}.{self.grid_search}.Disc1.f': 3.,
            f'{self.study_name}.{self.grid_search}.Disc1.g': 3.,
            f'{self.study_name}.{self.grid_search}.Disc1.h': 3.,
            f'{self.study_name}.{self.grid_search}.Disc1.j': 3.,

            # UQ
            # f'{self.study_name}.{self.grid_search}.samples_inputs_df': samples_inputs_df,
            # f'{self.study_name}.{self.grid_search}.samples_outputs_df': samples_outputs_df,
        }

        self.ee.load_study_from_input_dict(dict_values)

        self.ee.execute()

        grid_search_disc = self.ee.dm.get_disciplines_with_name(
            f'{self.study_name}.{self.grid_search}')[0]

        grid_search_disc_output = grid_search_disc.get_sosdisc_outputs()
        doe_disc_samples = grid_search_disc_output['samples_inputs_df']
        y_dict = grid_search_disc_output['GridSearch.Disc1.y_dict']

        print("bonjour")


if '__main__' == __name__:
    cls = TestUncertaintyQuantification()
    cls.setUp()
    unittest.main()
