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
from sos_trades_core.sos_processes.test.test_morphological_matrix.usecase_morphological_matrix import Study
from sos_trades_core.sos_processes.test.test_morphological_matrix_with_setup.usecase_morphological_matrix import Study as Study_with_setup


class TestMorphologicalMatrixEval(unittest.TestCase):
    """
    SoSMorphMatrixEval test class
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
        self.morph_matrix = 'MORPH_MATRIX'

    def test_01_morphological_matrix_eval(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_morphological_matrix')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()
        self.exec_eng.display_treeview_nodes()

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'b', 'name', 'x'])

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_outputs')['name'].values.tolist(), ['indicator', 'residuals_history', 'y', 'y_dict'])

        morph_matrix_eval_disc = self.exec_eng.dm.get_disciplines_with_name(
            'MyCase.MORPH_MATRIX')[0]

        # select eval_inputs 'a', 'x'
        eval_inputs = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')
        eval_inputs.loc[eval_inputs['name'] == 'a', [
            'selected_input', 'input_variable_name']] = [True, 'a_list']

        eval_inputs.loc[eval_inputs['name'] == 'x', [
            'selected_input', 'input_variable_name']] = [True, 'x_list']
        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.eval_inputs': eval_inputs}

        self.exec_eng.load_study_from_input_dict(dict_values)

        # check dynamic inputs 'a_list', 'x_list'
        self.assertListEqual(list(morph_matrix_eval_disc._data_in.keys()), [
                             'activation_morphological_matrix', 'selected_scenarios', 'eval_inputs', 'eval_outputs', 'linearization_mode', 'cache_type', 'cache_file_path', 'a_list', 'x_list'])

        # add 'name' and 'b' as selected eval input and remove 'a'
        eval_inputs = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')
        eval_inputs.loc[eval_inputs['name'] == 'name', [
            'selected_input', 'input_variable_name']] = [True, 'name_list']
        eval_inputs.loc[eval_inputs['name'] == 'b', [
            'selected_input', 'input_variable_name']] = [True, 'b_list']
        eval_inputs.loc[eval_inputs['name'] == 'a', [
            'selected_input']] = False

        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.eval_inputs': eval_inputs}

        self.exec_eng.load_study_from_input_dict(dict_values)

        # check 'name_list' dynamic inputs and 'a_list' removed
        self.assertListEqual(list(morph_matrix_eval_disc._data_in.keys()), [
                             'activation_morphological_matrix', 'selected_scenarios', 'eval_inputs', 'eval_outputs', 'linearization_mode', 'cache_type', 'cache_file_path', 'x_list', 'b_list', 'name_list'])

        # check type, possible_values and range of selected eval inputs
        self.assertListEqual(self.exec_eng.dm.get_data(
            'MyCase.MORPH_MATRIX.b_list', 'possible_values'), [0, 2, 5])
        self.assertListEqual(self.exec_eng.dm.get_data(
            'MyCase.MORPH_MATRIX.name_list', 'possible_values'), ['A1', 'A2', 'A3'])
        self.assertListEqual(self.exec_eng.dm.get_data(
            'MyCase.MORPH_MATRIX.x_list', 'range'), [1.0, 10.0])

        # add 'y' as selected eval output
        eval_outputs = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.morph_matrix}.eval_outputs')
        eval_outputs.loc[eval_outputs['name'] == 'y', [
            'selected_output', 'output_variable_name']] = [True, 'y_out']

        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.Disc1.a': 3,
            f'{self.study_name}.{self.morph_matrix}.Disc1.x_dict': {},
            f'{self.study_name}.{self.morph_matrix}.eval_outputs': eval_outputs,
            f'{self.study_name}.{self.morph_matrix}.x_list': [5., 5.5],
            f'{self.study_name}.{self.morph_matrix}.name_list': ['A1'],
            f'{self.study_name}.{self.morph_matrix}.b_list': [2]
        }

        self.exec_eng.load_study_from_input_dict(dict_values)

        activation_df = pd.DataFrame({'selected_scenario': False,
                                      'scenario_name': ['scenario_1', 'scenario_2'],
                                      'b_list': [2, 2],
                                      'name_list': ['A1', 'A1'],
                                      'x_list': [5.0, 5.5]})

        # check generated morphological matrix
        self.assertTrue(activation_df.equals(
            self.exec_eng.dm.get_value('MyCase.MORPH_MATRIX.activation_morphological_matrix')))
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.selected_scenarios').values.tolist(), [])

        self.exec_eng.execute()

        # check unchanged morphological matrix
        self.assertTrue(activation_df.equals(
            self.exec_eng.dm.get_value('MyCase.MORPH_MATRIX.activation_morphological_matrix')))

        # modifiy 'b_list' eval input values
        dict_values = {f'{self.study_name}.{self.morph_matrix}.b_list': [0, 2]}
        self.exec_eng.load_study_from_input_dict(dict_values)

        activation_df = pd.DataFrame({'selected_scenario': False,
                                      'scenario_name': ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4'],
                                      'b_list': [0, 0, 2, 2],
                                      'name_list': ['A1', 'A1', 'A1', 'A1'],
                                      'x_list': [5.0, 5.5, 5.0, 5.5]})

        # check generated morphological matrix
        self.assertTrue(activation_df.equals(
            self.exec_eng.dm.get_value('MyCase.MORPH_MATRIX.activation_morphological_matrix')))
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.selected_scenarios').values.tolist(), [])

        # desactivate scenarios
        activation_df = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.activation_morphological_matrix')
        activation_df['selected_scenario'] = [True, False, True, False]

        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.activation_morphological_matrix': activation_df
        }

        self.exec_eng.load_study_from_input_dict(dict_values)

        activation_df = pd.DataFrame({'selected_scenario': [True, False, True, False],
                                      'scenario_name': ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4'],
                                      'b_list': [0, 0, 2, 2],
                                      'name_list': ['A1', 'A1', 'A1', 'A1'],
                                      'x_list': [5.0, 5.5, 5.0, 5.5]})

        selected_scenarios_df = pd.DataFrame({'scenario_name': ['scenario_1', 'scenario_3'],
                                              'b_list': [0,  2],
                                              'name_list': ['A1', 'A1'],
                                              'x_list': [5.0,  5.0]})

        # check morphological matrix after desactivating scenarios
        self.assertTrue(activation_df.equals(
            self.exec_eng.dm.get_value('MyCase.MORPH_MATRIX.activation_morphological_matrix')))
        self.assertTrue(np.array_equal(selected_scenarios_df.values, self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.selected_scenarios').values))

        # rename scenarios
        activation_df = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.activation_morphological_matrix')
        activation_df['scenario_name'] = [
            'scen_A', 'scen_B', 'scen_C', 'scen_D']

        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.activation_morphological_matrix': activation_df
        }

        self.exec_eng.load_study_from_input_dict(dict_values)

        activation_df = pd.DataFrame({'selected_scenario': [True, False, True, False],
                                      'scenario_name': ['scen_A', 'scen_B', 'scen_C', 'scen_D'],
                                      'b_list': [0, 0, 2, 2],
                                      'name_list': ['A1', 'A1', 'A1', 'A1'],
                                      'x_list': [5.0, 5.5, 5.0, 5.5]})

        selected_scenarios_df = pd.DataFrame({'scenario_name': ['scen_A', 'scen_C'],
                                              'b_list': [0, 2],
                                              'name_list': ['A1', 'A1'],
                                              'x_list': [5.0, 5.0]})

        # check morphological matrix after renaming scenarios
        self.assertTrue(activation_df.equals(
            self.exec_eng.dm.get_value('MyCase.MORPH_MATRIX.activation_morphological_matrix')))
        self.assertTrue(np.array_equal(selected_scenarios_df.values, self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.selected_scenarios').values))

        # unselect 'name' as eval input
        eval_inputs = self.exec_eng.dm.get_value(
            f'{self.study_name}.{self.morph_matrix}.eval_inputs')
        eval_inputs.loc[eval_inputs['name'] == 'name', [
            'selected_input']] = [False]
        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.eval_inputs': eval_inputs}

        self.exec_eng.load_study_from_input_dict(dict_values)

        activation_df = pd.DataFrame({'selected_scenario': False,
                                      'scenario_name': ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4'],
                                      'b_list': [0, 0, 2, 2],
                                      'x_list': [5.0, 5.5, 5.0, 5.5]})

        # check generated morphological matrix
        self.assertTrue(activation_df.equals(
            self.exec_eng.dm.get_value('MyCase.MORPH_MATRIX.activation_morphological_matrix')))
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.selected_scenarios').values.tolist(), [])

        # modify 'x_list' value
        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.x_list': [5.0, 5.5, 6.0]}

        self.exec_eng.load_study_from_input_dict(dict_values)

        activation_df = pd.DataFrame({'selected_scenario': False,
                                      'scenario_name': ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'scenario_6'],
                                      'b_list': [0, 0, 0, 2, 2, 2],
                                      'x_list': [5.0, 5.5, 6.0, 5.0, 5.5, 6.0]})

        # check generated morphological matrix
        self.assertTrue(activation_df.equals(
            self.exec_eng.dm.get_value('MyCase.MORPH_MATRIX.activation_morphological_matrix')))
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.selected_scenarios').values.tolist(), [])

        # desactivate scenarios
        activation_df = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.activation_morphological_matrix')
        activation_df['selected_scenario'] = [
            True, False, False, True, True, True]

        dict_values = {
            f'{self.study_name}.{self.morph_matrix}.activation_morphological_matrix': activation_df
        }

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.exec_eng.execute()

        # check morphological amtrix after desactivating scenarios
        self.assertTrue(activation_df.equals(
            self.exec_eng.dm.get_value('MyCase.MORPH_MATRIX.activation_morphological_matrix')))

        selected_scenarios_df = pd.DataFrame({'scenario_name': ['scenario_1', 'scenario_4', 'scenario_5', 'scenario_6'],
                                              'b_list': [0, 2, 2, 2],
                                              'x_list': [5.0, 5.0, 5.5, 6.0]})
        self.assertTrue(np.array_equal(selected_scenarios_df.values, self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.selected_scenarios').values))

        a = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.Disc1.a')
        y_out = {'scenario_1': a * activation_df.loc[activation_df['scenario_name'] == 'scenario_1',
                                                     'x_list'].values[0] + activation_df.loc[activation_df['scenario_name'] == 'scenario_1', 'b_list'].values[0],
                 'scenario_4': a * activation_df.loc[activation_df['scenario_name'] == 'scenario_4',
                                                     'x_list'].values[0] + activation_df.loc[activation_df['scenario_name'] == 'scenario_4', 'b_list'].values[0],
                 'scenario_5': a * activation_df.loc[activation_df['scenario_name'] == 'scenario_5',
                                                     'x_list'].values[0] + activation_df.loc[activation_df['scenario_name'] == 'scenario_5', 'b_list'].values[0],
                 'scenario_6': a * activation_df.loc[activation_df['scenario_name'] == 'scenario_6',
                                                     'x_list'].values[0] + activation_df.loc[activation_df['scenario_name'] == 'scenario_6', 'b_list'].values[0]}

        # check eval output value
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.y_out'), y_out)

    def test_02_morphological_matrix_eval_from_process(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_morphological_matrix')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()

        # import usecase associated to test_morphological_matrix process
        usecase = Study(execution_engine=self.exec_eng)
        usecase.study_name = self.study_name
        dict_values = usecase.setup_usecase()
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        activation_df = pd.DataFrame({'selected_scenario': [True, False, False, True],
                                      'scenario_name': ['scenario_A', 'scenario_B', 'scenario_C', 'scenario_D'],
                                      'b_list': [0, 0, 2, 2],
                                      'x_list': [5.0, 5.5, 5.0, 5.5]})

        eval_inputs = pd.DataFrame({'selected_input': [False, True, False, True],
                                    'name': ['a', 'b', 'name', 'x'],
                                    'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1'],
                                    'input_variable_name': ['', 'b_list', '', 'x_list']})

        eval_outputs = pd.DataFrame({'selected_output': [False, False, True, False],
                                     'name': ['indicator', 'residuals_history', 'y', 'y_dict'],
                                     'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1'],
                                     'output_variable_name': ['', '', 'y_out', '']})

        # check input values
        self.assertTrue(eval_inputs.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')))
        self.assertTrue(eval_outputs.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_outputs')))
        self.assertTrue(activation_df.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.activation_morphological_matrix')))

        a = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.Disc1.a')
        y_out = {'scenario_A': a * activation_df.loc[activation_df['scenario_name'] == 'scenario_A',
                                                     'x_list'].values[0] + activation_df.loc[activation_df['scenario_name'] == 'scenario_A', 'b_list'].values[0],
                 'scenario_D': a * activation_df.loc[activation_df['scenario_name'] == 'scenario_D',
                                                     'x_list'].values[0] + activation_df.loc[activation_df['scenario_name'] == 'scenario_D', 'b_list'].values[0]}

        # check output values
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.y_out'), y_out)

    def test_03_morphological_matrix_from_process_2(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_morphological_matrix')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()

        # import usecase associated to test_morphological_matrix process
        usecase = Study(execution_engine=self.exec_eng)
        usecase.study_name = self.study_name
        dict_values = usecase.setup_usecase()

        eval_inputs_short = pd.DataFrame({'selected_input': [True, True],
                                          'name': ['b', 'x'],
                                          'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1'],
                                          'input_variable_name': ['b_list',  'x_list']})

        eval_outputs_short = pd.DataFrame({'selected_output': [True],
                                           'name': ['y'],
                                           'namespace': [f'{self.morph_matrix}.Disc1'],
                                           'output_variable_name': ['y_out']})

        activation_df = pd.DataFrame({'selected_scenario': [True, False, False, True],
                                      'scenario_name': ['scenario_A', 'scenario_2', 'scenario_3', 'scenario_D'],
                                      'b_list': [0, 0, 2, 2],
                                      'x_list': [5.0, 5.5, 5.0, 5.5]})

        dict_values.update(
            {f'{self.study_name}.{self.morph_matrix}.eval_inputs': eval_inputs_short,
             f'{self.study_name}.{self.morph_matrix}.eval_outputs': eval_outputs_short,
             f'{self.study_name}.{self.morph_matrix}.activation_morphological_matrix': activation_df})
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        eval_inputs = pd.DataFrame({'selected_input': [False, True, False, True],
                                    'name': ['a', 'b', 'name', 'x'],
                                    'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1'],
                                    'input_variable_name': ['', 'b_list', '', 'x_list']})

        eval_outputs = pd.DataFrame({'selected_output': [False, False, True, False],
                                     'name': ['indicator', 'residuals_history', 'y', 'y_dict'],
                                     'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1'],
                                     'output_variable_name': ['', '', 'y_out', '']})

        # check input values
        self.assertTrue(eval_inputs.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')))
        self.assertTrue(eval_outputs.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_outputs')))
        self.assertTrue(activation_df.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.activation_morphological_matrix')))

        a = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.Disc1.a')
        y_out = {'scenario_A': a * activation_df.loc[activation_df['scenario_name'] == 'scenario_A',
                                                     'x_list'].values[0] + activation_df.loc[activation_df['scenario_name'] == 'scenario_A', 'b_list'].values[0],
                 'scenario_D': a * activation_df.loc[activation_df['scenario_name'] == 'scenario_D',
                                                     'x_list'].values[0] + activation_df.loc[activation_df['scenario_name'] == 'scenario_D', 'b_list'].values[0]}

        # check output values
        self.assertDictEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.y_out'), y_out)

    def test_04_morphological_matrix_eval_usecase_with_setup_sos_discipline(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_morphological_matrix_with_setup')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'b', 'x'])

        self.exec_eng.load_study_from_input_dict(
            {'MyCase.MORPH_MATRIX.AC_list': ['AC1', 'AC2']})

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'AC1.dyn_input_1', 'AC2.dyn_input_1', 'b', 'x'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')['namespace'].values.tolist(), [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_outputs')['name'].values.tolist(), ['AC1.dyn_output', 'AC2.dyn_output', 'indicator', 'residuals_history', 'y'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_outputs')['namespace'].values.tolist(), [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', f'{self.morph_matrix}'])

        # import usecase associated to test_morphological_matrix process
        usecase = Study_with_setup(execution_engine=self.exec_eng)
        usecase.study_name = self.study_name
        dict_values = usecase.setup_usecase()
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        activation_df = pd.DataFrame({'selected_scenario': [True, False, False, True],
                                      'scenario_name': ['scenario_A', 'scenario_B', 'scenario_C', 'scenario_D'],
                                      'AC1_dyn_input_list': [1.0, 1.0, 3.0, 3.0, ],
                                      'b_list': [0.0, 2.0, 0.0, 2.0], })

        eval_outputs = pd.DataFrame({'selected_output': [True, False, False, False, True],
                                     'name': ['AC1.dyn_output', 'AC2.dyn_output', 'indicator', 'residuals_history', 'y'],
                                     'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', f'{self.morph_matrix}'],
                                     'output_variable_name': ['dyn_outputs_out', '', '', '', 'y_out']})
        eval_inputs = pd.DataFrame({'selected_input': [False, True, False, True, False],
                                    'name': ['a', 'AC1.dyn_input_1', 'AC2.dyn_input_1', 'b', 'x'],
                                    'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1',
                                                  f'{self.morph_matrix}.Disc1',
                                                  f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', ],
                                    'input_variable_name': ['', 'AC1_dyn_input_list', '', 'b_list', '']})
        # check input values
        self.assertTrue(eval_inputs.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')))
        self.assertTrue(eval_outputs.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_outputs')))
        self.assertTrue(activation_df.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.activation_morphological_matrix')))

        dyn_output_out = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.dyn_outputs_out')

        dyn_output_out_ref = {'scenario_A': activation_df.loc[activation_df['scenario_name'] == 'scenario_A',
                                                              'AC1_dyn_input_list'].values[0]**2,
                              'scenario_D': activation_df.loc[activation_df['scenario_name'] == 'scenario_D',
                                                              'AC1_dyn_input_list'].values[0]**2, }
        self.assertDictEqual(dyn_output_out, dyn_output_out_ref)

    def test_05_morphological_matrix_eval_usecase_with_setup_sos_discipline_direct(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_morphological_matrix_with_setup')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()

        # import usecase associated to test_morphological_matrix process
        usecase = Study_with_setup(execution_engine=self.exec_eng)
        usecase.study_name = self.study_name
        dict_values = usecase.setup_usecase()
        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        activation_df = pd.DataFrame({'selected_scenario': [True, False, False, True],
                                      'scenario_name': ['scenario_A', 'scenario_B', 'scenario_C', 'scenario_D'],
                                      'AC1_dyn_input_list': [1.0, 1.0, 3.0, 3.0, ],
                                      'b_list': [0.0, 2.0, 0.0, 2.0], })

        eval_outputs = pd.DataFrame({'selected_output': [True, False, False, False, True],
                                     'name': ['AC1.dyn_output', 'AC2.dyn_output', 'indicator', 'residuals_history', 'y'],
                                     'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', f'{self.morph_matrix}'],
                                     'output_variable_name': ['dyn_outputs_out', '', '', '', 'y_out']})
        eval_inputs = pd.DataFrame({'selected_input': [False, True, False, True, False],
                                    'name': ['a', 'AC1.dyn_input_1', 'AC2.dyn_input_1', 'b', 'x'],
                                    'namespace': [f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}.Disc1',
                                                  f'{self.morph_matrix}.Disc1',
                                                  f'{self.morph_matrix}.Disc1', f'{self.morph_matrix}', ],
                                    'input_variable_name': ['', 'AC1_dyn_input_list', '', 'b_list', '']})
        # check input values
        self.assertTrue(eval_inputs.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_inputs')))
        self.assertTrue(eval_outputs.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.eval_outputs')))
        self.assertTrue(activation_df.equals(self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.activation_morphological_matrix')))

        dyn_output_out = self.exec_eng.dm.get_value(
            'MyCase.MORPH_MATRIX.dyn_outputs_out')

        dyn_output_out_ref = {'scenario_A': activation_df.loc[activation_df['scenario_name'] == 'scenario_A',
                                                              'AC1_dyn_input_list'].values[0]**2,
                              'scenario_D': activation_df.loc[activation_df['scenario_name'] == 'scenario_D',
                                                              'AC1_dyn_input_list'].values[0]**2, }
        self.assertDictEqual(dyn_output_out, dyn_output_out_ref)

    def test_06_multi_scenario_of_morphological_matrix_eval(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_multi_scenario_morphological_matrix')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()

        dict_values = {}
        dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = [
            'scenario_1', 'scenario_2']

        ns_scenario_1_disc_1 = f'multi_scenarios.scenario_1.MORPH_MATRIX.Disc1'
        ns_scenario_2_disc_1 = f'multi_scenarios.scenario_2.MORPH_MATRIX.Disc1'

        eval_inputs_scenario_1 = pd.DataFrame({'selected_input': [True, True, False, False],
                                               'name': ['a', 'b', 'name', 'x'],
                                               'namespace': [ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1],
                                               'input_variable_name': ['a_list', 'b_list', '', '']})

        eval_inputs_scenario_2 = pd.DataFrame({'selected_input': [False, False, False, True],
                                               'name': ['a', 'b', 'name', 'x'],
                                               'namespace': [ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1],
                                               'input_variable_name': ['', '', '', 'x_list']})

        eval_outputs_scenario_1 = pd.DataFrame({'selected_output': [False, False, True, False],
                                                'name': ['indicator', 'residuals_history', 'y', 'y_dict'],
                                                'namespace': [ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1],
                                                'output_variable_name': ['', '', 'y_out', '']})

        eval_outputs_scenario_2 = pd.DataFrame({'selected_output': [False, False, True, False],
                                                'name': ['indicator', 'residuals_history', 'y', 'y_dict'],
                                                'namespace': [ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1],
                                                'output_variable_name': ['', '', 'y_out', '']})

        # set eval_inputs
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.eval_inputs'] = eval_inputs_scenario_1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.eval_inputs'] = eval_inputs_scenario_2
        # set eval_outputs
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.eval_outputs'] = eval_outputs_scenario_1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.eval_outputs'] = eval_outputs_scenario_2
        # set eval_inputs values list
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.a_list'] = [
            0, 2]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.b_list'] = [
            2, 5]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.x_list'] = [
            5.0, 6.0, 7.0]
        # set other inputs
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.Disc1.x'] = 5.0
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.Disc1.name'] = 'A1'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.Disc1.a'] = 1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.Disc1.b'] = 2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.Disc1.name'] = 'A1'
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.activation_morphological_matrix'] = pd.DataFrame({'selected_scenario': [True, True, True, True],
                                                                                                                                  'scenario_name': ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4'],
                                                                                                                                  'a_list': [0, 0, 2, 2],
                                                                                                                                  'b_list': [2, 5, 2, 5]})
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.activation_morphological_matrix'] = pd.DataFrame({'selected_scenario': [True, True, True],
                                                                                                                                  'scenario_name': ['scenario_1', 'scenario_2', 'scenario_3'],
                                                                                                                                  'x_list': [5.0, 6.0, 7.0]})

        self.exec_eng.load_study_from_input_dict(dict_values)

        activation_df_scenario_1 = pd.DataFrame({'selected_scenario': True,
                                                 'scenario_name': ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4'],
                                                 'a_list': [0, 0, 2, 2],
                                                 'b_list': [2, 5, 2, 5]})

        activation_df_scenario_2 = pd.DataFrame({'selected_scenario': True,
                                                 'scenario_name': ['scenario_1', 'scenario_2', 'scenario_3'],
                                                 'x_list': [5.0, 6.0, 7.0]})

        self.assertTrue(activation_df_scenario_1.equals(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.activation_morphological_matrix')))
        self.assertTrue(activation_df_scenario_2.equals(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.activation_morphological_matrix')))

        # run with all the scenarios activated
        self.exec_eng.execute()

        self.assertDictEqual(
            self.exec_eng.dm.get_value(f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.y_out'), {'scenario_1': 2.0, 'scenario_2': 5.0, 'scenario_3': 12.0, 'scenario_4': 15.0})
        self.assertDictEqual(
            self.exec_eng.dm.get_value(
                f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.y_out'),
            {'scenario_1': 7.0, 'scenario_2': 8.0, 'scenario_3': 9.0})

        # scenario activation in activation_morphological_matrix
        activation_df_scenario_1 = pd.DataFrame({'selected_scenario': [True, False, False, True],
                                                 'scenario_name': ['scenario_ab1', 'scenario_ab2', 'scenario_ab3', 'scenario_ab4'],
                                                 'a_list': [0, 0, 2, 2],
                                                 'b_list': [2, 5, 2, 5]})

        activation_df_scenario_2 = pd.DataFrame({'selected_scenario': [True, True, True],
                                                 'scenario_name': ['scenario_x5', 'scenario_x6', 'scenario_x7'],
                                                 'x_list': [5.0, 6.0, 7.0]})

        dict_values = {
            f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.activation_morphological_matrix': activation_df_scenario_1,
            f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.activation_morphological_matrix': activation_df_scenario_2}

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.display_treeview_nodes()
        self.exec_eng.execute()

        # check all discipline status DONE
        for disc_id in self.exec_eng.dm.disciplines_dict.keys():
            self.assertEqual(self.exec_eng.dm.get_discipline(
                disc_id).status, 'DONE')

        # check eval_outputs content
        y_out_scenario_1 = {'scenario_ab1': 2.0, 'scenario_ab4': 15.0}
        y_out_scenario_2 = {'scenario_x5': 7.0,
                            'scenario_x6': 8.0, 'scenario_x7': 9.0}
        self.assertDictEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.y_out'), y_out_scenario_1)
        self.assertDictEqual(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.y_out'),
            y_out_scenario_2)

        # check slected_scenarios inputs for each scenario
        selected_scenarios_1 = pd.DataFrame({'scenario_name': ['scenario_ab1', 'scenario_ab4'],
                                             'a_list': [0, 2],
                                             'b_list': [2, 5]})
        self.assertTrue(np.array_equal(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.selected_scenarios').values, selected_scenarios_1.values))

        selected_scenarios_2 = pd.DataFrame({'scenario_name': ['scenario_x5', 'scenario_x6', 'scenario_x7'],
                                             'x_list': [5.0, 6.0, 7.0]})
        self.assertTrue(np.array_equal(self.exec_eng.dm.get_value(
            f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.selected_scenarios').values, selected_scenarios_2.values))

    def test_07_multi_scenario_of_morphological_matrix_eval_with_setup_sos_discipline(self):

        sa_builder = self.exec_eng.factory.get_builder_from_process(
            self.repo, 'test_multi_scenario_morphological_matrix_with_setup')

        self.exec_eng.factory.set_builders_to_coupling_builder(
            sa_builder)

        self.exec_eng.configure()

        dict_values = {}
        scenario_list = ['scenario_1', 'scenario_2']
        dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = scenario_list
        for scenario in scenario_list:
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.MORPH_MATRIX.x'] = 3.
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.MORPH_MATRIX.Disc1.a'] = 1
            dict_values[f'{self.study_name}.multi_scenarios.{scenario}.MORPH_MATRIX.Disc1.b'] = 0.

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'b', 'x'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'b', 'x'])

        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.AC_list'] = [
            'AC1', 'AC2', 'AC3']
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.AC_list'] = [
            'AC4']

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'AC1.dyn_input_1', 'AC2.dyn_input_1', 'AC3.dyn_input_1', 'b', 'x'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'AC4.dyn_input_1', 'b', 'x'])

        ns_scenario_1_disc_1 = 'multi_scenarios.scenario_1.MORPH_MATRIX.Disc1'
        eval_inputs_scenario_1 = pd.DataFrame({'selected_input': [True, True, False, False, False, False, False],
                                               'name': ['a', 'AC1.dyn_input_1', 'AC2.dyn_input_1', 'AC3.dyn_input_1', 'b', 'name', 'x'],
                                               'namespace': [ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, 'multi_scenarios.scenario_1.MORPH_MATRIX'],
                                               'input_variable_name': ['a_list', 'ac1_list', '', '', '', '', '']})

        ns_scenario_2_disc_1 = 'multi_scenarios.scenario_2.MORPH_MATRIX.Disc1'
        eval_inputs_scenario_2 = pd.DataFrame({'selected_input': [False, True, False, False, True],
                                               'name': ['a', 'AC4.dyn_input_1', 'b', 'name', 'x'],
                                               'namespace': [ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1, ns_scenario_2_disc_1, 'multi_scenarios.scenario_2.MORPH_MATRIX'],
                                               'input_variable_name': ['', 'ac2_list', '', '', 'x_list']})

        eval_outputs_scenario_1 = pd.DataFrame({'selected_output': [False, False, False, False, False, True],
                                                'name': ['AC1.dyn_output', 'AC2.dyn_output', 'AC3.dyn_output', 'indicator', 'residuals_history', 'y'],
                                                'namespace': [ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, ns_scenario_1_disc_1, 'multi_scenarios.scenario_1.MORPH_MATRIX', 'multi_scenarios.scenario_1.MORPH_MATRIX'],
                                                'output_variable_name': ['', '', '', '', '', 'y_out']})

        eval_outputs_scenario_2 = pd.DataFrame({'selected_output': [False, False, False, True],
                                                'name': ['AC4.dyn_output', 'indicator', 'residuals_history', 'y'],
                                                'namespace': [ns_scenario_2_disc_1, ns_scenario_2_disc_1, 'multi_scenarios.scenario_2.MORPH_MATRIX', 'multi_scenarios.scenario_2.MORPH_MATRIX'],
                                                'output_variable_name': ['', '', '', 'y_out']})

        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.eval_inputs'] = eval_inputs_scenario_1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.eval_inputs'] = eval_inputs_scenario_2
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.eval_outputs'] = eval_outputs_scenario_1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.eval_outputs'] = eval_outputs_scenario_2

        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.a_list'] = [
            1, 2]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.ac1_list'] = [
            0., 3.]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.ac2_list'] = [
            10., 15.]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.x_list'] = [
            30., 35.]

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        # check Disc1 discipline not duplicated in treeview
        disc_1_node = self.exec_eng.get_treeview(
        ).root.children[0].children[0].children[0].children[0]
        self.assertEqual(self.exec_eng.dm.get_discipline(
            disc_1_node.disc_ids[0]).sos_name, 'Disc1')
        self.assertEqual(len(disc_1_node.disc_ids), 1)

        # check all discipline status DONE
        for disc_id in self.exec_eng.dm.disciplines_dict.keys():
            self.assertEqual(self.exec_eng.dm.get_discipline(
                disc_id).status, 'DONE')

        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.AC_list'] = [
            'AC1', 'AC2']
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.AC_list'] = [
            'AC3', 'AC4']

        self.exec_eng.load_study_from_input_dict(dict_values)

        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'AC1.dyn_input_1', 'AC2.dyn_input_1', 'b', 'x'])
        self.assertListEqual(self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_2.MORPH_MATRIX.eval_inputs')['name'].values.tolist(), ['a', 'AC3.dyn_input_1', 'AC4.dyn_input_1', 'b', 'x'])

        self.exec_eng.execute()

        # check all discipline status DONE
        for disc_id in self.exec_eng.dm.disciplines_dict.keys():
            self.assertEqual(self.exec_eng.dm.get_discipline(
                disc_id).status, 'DONE')

        self.exec_eng.execute()

        # check all discipline status DONE after reexecution
        for disc_id in self.exec_eng.dm.disciplines_dict.keys():
            self.assertEqual(self.exec_eng.dm.get_discipline(
                disc_id).status, 'DONE')

        activation_df_scenario_1 = self.exec_eng.dm.get_value(
            'MyCase.multi_scenarios.scenario_1.MORPH_MATRIX.activation_morphological_matrix')
        activation_df_scenario_1['selected_scenario'] = False
        dict_values['MyCase.multi_scenarios.scenario_1.MORPH_MATRIX.activation_morphological_matrix'] = activation_df_scenario_1

        self.exec_eng.load_study_from_input_dict(dict_values)
        self.exec_eng.execute()

        # check all discipline status DONE without any scenario activated
        for disc_id in self.exec_eng.dm.disciplines_dict.keys():
            self.assertEqual(self.exec_eng.dm.get_discipline(
                disc_id).status, 'DONE')


if '__main__' == __name__:
    cls = TestMorphologicalMatrixEval()
    cls.setUp()
    cls.test_01_morphological_matrix_eval()
