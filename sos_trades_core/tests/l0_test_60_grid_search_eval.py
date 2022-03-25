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
import numpy as np
from pandas.testing import assert_frame_equal

from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.sos_wrapping.analysis_discs.grid_search_eval import GridSearchEval
from sos_trades_core.execution_engine.sos_coupling import SoSCoupling


class TestGridSearchEval(unittest.TestCase):
    """
    SoSGridSearchEval test class
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.repo = 'sos_trades_core.sos_processes.test'
        self.base_path = 'sos_trades_core.sos_wrapping.test_discs'
        self.exec_eng = ExecutionEngine(self.namespace)
        self.factory = self.exec_eng.factory
        self.grid_search = 'GridSearch'
        self.proc_name = 'test_grid_search'

    def test_01_grid_search_eval(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, self.proc_name)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        print('Study first configure!')

        self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.eval_inputs')
        # self.exec_eng.dm.get_data('MyCase.GridSearch.eval_inputs')[
        #     'possible_values']

        # dict_values = {}
        # self.exec_eng.load_study_from_input_dict(dict_values)

        eval_inputs = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.eval_inputs')
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.x', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.j', ['selected_input']] = True

        eval_outputs = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.eval_outputs')
        eval_outputs.loc[eval_outputs['full_name'] ==
                         f'{self.grid_search}.Disc1.y', ['selected_output']] = True

        dict_values = {
            # GRID SEARCH INPUTS
            f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
            f'{self.study_name}.{self.grid_search}.eval_outputs': eval_outputs,

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
        }

        self.exec_eng.load_study_from_input_dict(dict_values)

        ds = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.design_space')

        print(f'Second configure with design_space creation: \n {ds}')

        self.exec_eng.execute()

        grid_search_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.{self.grid_search}')[0]

        grid_search_disc_output = grid_search_disc.get_sosdisc_outputs()
        doe_disc_samples = grid_search_disc_output['doe_samples_dataframe']
        y_dict = grid_search_disc_output['GridSearch.Disc1.y_dict']
        ds = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.design_space')

        print(f'Study executed from the design_space: \n {ds}')
        print(f'Study executed with the samples: \n {doe_disc_samples}')
        print(f'Study generated the output: y_dict \n {y_dict}')

        dspace = pd.DataFrame({
            'shortest_name': ['x', 'j'],
            'lower_bnd': [5., 20.],
            'upper_bnd': [7., 25.],
            'nb_points': [3, 3],
            'full_name': ['GridSearch.Disc1.x', 'GridSearch.Disc1.j'],
        })

        dict_values = {
            f'{self.study_name}.{self.grid_search}.design_space': dspace,
        }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.configure()
        self.exec_eng.execute()

        grid_search_disc_output = grid_search_disc.get_sosdisc_outputs()

        doe_disc_samples = grid_search_disc_output['doe_samples_dataframe']
        y_dict = grid_search_disc_output['GridSearch.Disc1.y_dict']

        # CHECK THE GRIDSEARCH OUTPUTS
        doe_disc_samples_ref = pd.DataFrame({'scenario': [
                                            'scenario_1', 'scenario_2', 'scenario_3'], 'GridSearch.Disc1.x': [5.0, 6.0, 7.0]})
        y_dict_ref = {'scenario_1': 102.0,
                      'scenario_2': 122.0, 'scenario_3': 142.0}
        # assert_frame_equal(doe_disc_samples, doe_disc_samples_ref)
        # assert y_dict_ref == y_dict

        ds = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.design_space')

        print(f'Study executed from the design_space: \n {ds}')
        print(f'Study executed with the samples: \n {doe_disc_samples}')
        print(f'Study generated the output: y_dict \n {y_dict}')

        # TEST FOR 6 INPUTS
        eval_inputs = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.eval_inputs')
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.x', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.f', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.g', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.h', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.j', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.d', ['selected_input']] = True

        dict_values = {
            f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
            f'{self.study_name}.{self.grid_search}.eval_outputs': eval_outputs,
        }
        self.exec_eng.load_study_from_input_dict(dict_values)

        ds = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.design_space')

        print(f'Second configure with design_space creation: \n {ds}')

        self.exec_eng.execute()

        grid_search_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.{self.grid_search}')[0]

        grid_search_disc_output = grid_search_disc.get_sosdisc_outputs()
        doe_disc_samples = grid_search_disc_output['doe_samples_dataframe']
        y_dict = grid_search_disc_output['GridSearch.Disc1.y_dict']

        ds = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.design_space')

        print(f'Study executed from the design_space: \n {ds}')
        print(f'Study executed with the samples: \n {doe_disc_samples}')
        print(f'Study generated the output: y_dict \n {y_dict}')

        # CHANGE THE SELECTED INPUTS TO 2
        eval_inputs = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.eval_inputs')
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.x', ['selected_input']] = False
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.f', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.g', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.h', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.j', ['selected_input']] = False
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.d', ['selected_input']] = False

        dict_values = {
            f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
            f'{self.study_name}.{self.grid_search}.eval_outputs': eval_outputs,
        }
        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.dm.get_value(['MyCase.GridSearch.eval_inputs'][0])

        ds = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.design_space')

        print(f'Second configure with design_space creation: \n {ds}')

        self.exec_eng.execute()

        self.exec_eng.dm.get_value(['MyCase.GridSearch.eval_inputs'][0])

    def test_02_grid_search_shortest_name(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, self.proc_name)

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        print('Study first configure!')

        grid_search_disc = self.exec_eng.dm.get_disciplines_with_name(
            f'{self.study_name}.{self.grid_search}')[0]

        list = ['GridSearch.Disc1.d', 'GridSearch.Disc1.f', 'GridSearch.Disc1.g',
                'GridSearch.Disc1.h', 'GridSearch.Disc1.j', 'GridSearch.Disc1.x',
                'GridSearch.Disc2.d', 'GridSearch.Nana.Disc1.d', 'GridSearch.Nana.Disc2.d']

        shortest_list = grid_search_disc.generate_shortest_name(list)


if '__main__' == __name__:
    cls = TestGridSearchEval()
    cls.setUp()
    unittest.main()
