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

    def _test_01_grid_search_eval(self):

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
                        f'{self.grid_search}.Disc1.a', ['selected_input']] = True
        eval_inputs.loc[eval_inputs['full_name'] ==
                        f'{self.grid_search}.Disc1.x', ['selected_input']] = True

        eval_outputs = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.eval_outputs')
        eval_outputs.loc[eval_outputs['full_name'] ==
                         f'{self.grid_search}.Disc1.y', ['selected_output']] = True

        dict_values = {
            # f'{self.study_name}.{self.grid_search}.n_samples': n_samples,
            # f'{self.study_name}.{self.grid_search}.design_space': design_space,
            # f'{self.study_name}.{self.grid_search}.algo': algo,
            # f'{self.study_name}.{self.grid_search}.algo_options':
            # algo_options,
            # GRID SEARCH INPUTS
            f'{self.study_name}.{self.grid_search}.eval_inputs': eval_inputs,
            f'{self.study_name}.{self.grid_search}.eval_outputs': eval_outputs,

            # DISC1 INPUTS
            f'{self.study_name}.{self.grid_search}.Disc1.name': 'A1',
            f'{self.study_name}.{self.grid_search}.Disc1.a': 20,
            f'{self.study_name}.{self.grid_search}.Disc1.b': 2,
            f'{self.study_name}.{self.grid_search}.Disc1.x': 3.,
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
        y_dict = grid_search_disc_output['MyCase.GridSearch.Disc1.y_dict']
        ds = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.design_space')

        print(f'Study executed from the design_space: \n {ds}')
        print(f'Study executed with the samples: \n {doe_disc_samples}')
        print(f'Study generated the output: y_dict \n {y_dict}')

        dspace = pd.DataFrame({'variable': ['GridSearch.Disc1.x'],
                               'lower_bnd': [5.],
                               'upper_bnd': [7.],
                               'nb_points': [3],
                               })

        dict_values = {
            f'{self.study_name}.{self.grid_search}.design_space': dspace,
        }

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        grid_search_disc_output = grid_search_disc.get_sosdisc_outputs()

        doe_disc_samples = grid_search_disc_output['doe_samples_dataframe']
        y_dict = grid_search_disc_output['MyCase.GridSearch.Disc1.y_dict']
        ds = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.grid_search}.design_space')

        print(f'Study executed from the design_space: \n {ds}')
        print(f'Study executed with the samples: \n {doe_disc_samples}')
        print(f'Study generated the output: y_dict \n {y_dict}')


if '__main__' == __name__:
    cls = TestGridSearchEval()
    cls.setUp()
    cls.test_09_morphological_matrix_eval_of_scatter_discipline()
