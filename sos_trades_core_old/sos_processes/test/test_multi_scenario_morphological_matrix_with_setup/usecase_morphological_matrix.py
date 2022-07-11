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
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
from sos_trades_core.study_manager.study_manager import StudyManager
import pandas as pd


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):

        dict_values = {}
        scenario_list = ['scenario_1', 'scenario_2']
        dict_values[f'{self.study_name}.multi_scenarios.scenario_list'] = scenario_list

        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.AC_list'] = [
            'AC1', 'AC2', 'AC3']
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.AC_list'] = [
            'AC4']

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
            1, 2, 3]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.ac1_list'] = [
            0, 4]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.ac2_list'] = [
            10, 15]
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.x_list'] = [
            30, 35]

        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.x'] = 3
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.Disc1.a'] = 1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_1.MORPH_MATRIX.Disc1.b'] = 0
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.Disc1.a'] = 1
        dict_values[f'{self.study_name}.multi_scenarios.scenario_2.MORPH_MATRIX.Disc1.b'] = 0

        return dict_values


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.run()
